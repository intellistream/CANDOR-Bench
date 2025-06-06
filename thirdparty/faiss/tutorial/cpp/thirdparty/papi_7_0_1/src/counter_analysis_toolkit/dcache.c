#include "papi.h"
#include "caches.h"
#include "timing_kernels.h"
#include "dcache.h"
#include "params.h"
#include <math.h>

extern char* eventname;

int min_size, max_size, is_core = 0;

void d_cache_driver(char* papi_event_name, cat_params_t params, hw_desc_t *hw_desc, int latency_only, int mode)
{
    int pattern = 3;
    int stride, f, cache_line;
    int status, evtCode, test_cnt = 0;
    float ppb = 16;
    FILE *ofp_papi;
    char *sufx, *papiFileName;

    // Use component ID to check if event is a core event.
    if( strcmp(papi_event_name, "cat::latencies") && PAPI_OK != (status = PAPI_event_name_to_code(papi_event_name, &evtCode)) ) {
        error_handler(status, __LINE__);
    } else {
        if( 0 == PAPI_get_event_component(evtCode) )
            is_core = 1;
    }

    // Open file (pass handle to d_cache_test()).
    if(CACHE_READ_WRITE == mode){
        sufx = strdup(".data.writes");
    }else{
        sufx = strdup(".data.reads");
    }

    int l = strlen(params.outputdir)+strlen(papi_event_name)+strlen(sufx);
    papiFileName = (char *)calloc( 1+l, sizeof(char) );
    if (!papiFileName) {
        fprintf(stderr, "Unable to allocate memory. Skipping event %s.\n", papi_event_name);
        goto error0;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", params.outputdir, papi_event_name, sufx))) {
        fprintf(stderr, "sprintf error. Skipping event %s.\n", papi_event_name);
        goto error1;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Unable to open file %s. Skipping event %s.\n", papiFileName, papi_event_name);
        goto error1;
    }

    if( (NULL==hw_desc) || (0==hw_desc->dcache_line_size[0]) )
        cache_line = 64;
    else
        cache_line = hw_desc->dcache_line_size[0];

    // Print the core to which each thread is pinned.
    print_core_affinities(ofp_papi);

    // Go through each parameter variant.
    for(pattern = 3; pattern <= 4; ++pattern)
    {
        for(f = 1; f <= 2; f *= 2)
        {
            stride = cache_line*f;
            // PPB variation only makes sense if the pattern is not sequential.
            if(pattern != 4) 
            {
                for(ppb = 64; ppb >= 16; ppb -= 48)
                {
                    if( params.show_progress )
                    {
                        printf("%3d%%\b\b\b\b",(100*test_cnt++)/6);
                        fflush(stdout);
                    }
                    status = d_cache_test(pattern, params.max_iter, hw_desc, stride, ppb, papi_event_name, latency_only, mode, ofp_papi);
                    if( status < 0 )
                        goto error2;
                }
            }
            else
            {
                if( params.show_progress )
                {
                    printf("%3d%%\b\b\b\b",(100*test_cnt++)/6);
                    fflush(stdout);
                }
                status = d_cache_test(pattern, params.max_iter, hw_desc, stride, ppb, papi_event_name, latency_only, mode, ofp_papi);
                if( status < 0 )
                    goto error2;
            }
        }
    }
error2:
    if( params.show_progress )
    {
        size_t i;
        printf("100%%");
        for(i=0; i<strlen("Total:100%  Current test:100%"); i++) putchar('\b');
        fflush(stdout);
    }
 
    // Close files and free memory.
    fclose(ofp_papi);
error1:
    free(papiFileName);
error0:
    free(sufx);

    return;
}

int d_cache_test(int pattern, int max_iter, hw_desc_t *hw_desc, int stride_in_bytes, float pages_per_block, char* papi_event_name, int latency_only, int mode, FILE* ofp){
    int i,j,k;
    int *values;
    double ***rslts, *sorted_rslts;
    double ***counter, *sorted_counter;
    int status=0, guessCount, ONT;

    min_size = 2*1024/sizeof(uintptr_t);        // 2KB
    max_size = 1024*1024*1024/sizeof(uintptr_t);// 1GB

    // The number of different sizes we will guess, trying to find the right size.
    guessCount = 0;
    if( (NULL==hw_desc) || (hw_desc->cache_levels<=0) ){
        for(i=min_size; i<max_size; i*=2){
            // += 4 for i, i*1.25, i*1.5, i*1.75
            guessCount += 4;
        }
    }else{
        int numHier = hw_desc->cache_levels+1;
        for(j=0; j<numHier; ++j) {
            guessCount += hw_desc->pts_per_reg[j] + 1;
        }
        guessCount++; // To include endpoint.
    }

    // Get the number of threads.
    ONT = get_thread_count();

    // Latency results from the benchmark.
    rslts = (double ***)malloc(max_iter*sizeof(double **));
    for(i=0; i<max_iter; ++i){
        rslts[i] = (double **)malloc(guessCount*sizeof(double*));
        for(j=0; j<guessCount; ++j){
            rslts[i][j] = (double *)malloc(ONT*sizeof(double));
        }
    }
    sorted_rslts = (double *)malloc(max_iter*sizeof(double));

    // Counter results from the benchmark.
    counter = (double ***)malloc(max_iter*sizeof(double **));
    for(i=0; i<max_iter; ++i){
        counter[i] = (double **)malloc(guessCount*sizeof(double*));
        for(j=0; j<guessCount; ++j){
            counter[i][j] = (double *)malloc(ONT*sizeof(double));
        }
    }
    sorted_counter = (double *)malloc(max_iter*sizeof(double));

    // List of buffer sizes which are used in the benchmark.
    values = (int *)malloc(guessCount*sizeof(int));

    // Set the name of the event to be monitored during the benchmark.
    eventname = papi_event_name;

    for(i=0; i<max_iter; ++i){
        status = varyBufferSizes(values, rslts[i], counter[i], hw_desc, stride_in_bytes, pages_per_block, pattern, latency_only, mode, ONT);
        if( status < 0 )
            goto cleanup;
    }

    // Sort and print latency and counter results.
    fprintf(ofp, "# PTRN=%d, STRIDE=%d, PPB=%f, ThreadCount=%d\n", pattern, stride_in_bytes, pages_per_block, ONT);

    if(latency_only) {

        for(j=0; j<guessCount; ++j){
            fprintf(ofp, "%d", values[j]);
            for(k=0; k<ONT; ++k){
                for(i=0; i<max_iter; ++i){
                    sorted_rslts[i] = rslts[i][j][k];
                }
                qsort(sorted_rslts, max_iter, sizeof(double), compar_lf);
                fprintf(ofp, " %.4lf", sorted_rslts[0]);
            }
            fprintf(ofp, "\n");
        }

    } else {

        for(j=0; j<guessCount; ++j){
            fprintf(ofp, "%d", values[j]);
            for(k=0; k<ONT; ++k){
                for(i=0; i<max_iter; ++i){
                    sorted_counter[i] = counter[i][j][k];
                }
                qsort(sorted_counter, max_iter, sizeof(double), compar_lf);
                fprintf(ofp, " %lf", sorted_counter[0]);
            }
            fprintf(ofp, "\n");
        }
    }

cleanup:
    for(i=0; i<max_iter; ++i){
        for(j=0; j<guessCount; ++j){
            free(rslts[i][j]);
            free(counter[i][j]);
        }
        free(rslts[i]);
        free(counter[i]);
    }
    free(rslts);
    free(counter);
    free(sorted_rslts);
    free(sorted_counter);
    free(values);

    return status;
}


int varyBufferSizes(int *values, double **rslts, double **counter, hw_desc_t *hw_desc, int stride_in_bytes, float pages_per_block, int pattern, int latency_only, int mode, int ONT){
    int i, j, k, cnt;
    long active_buf_len;
    int allocErr = 0;
    run_output_t out;

    int stride = stride_in_bytes/sizeof(uintptr_t);

    uintptr_t rslt=42, *v[ONT], *ptr[ONT];

    // Allocate memory for each thread to traverse.
    //#pragma omp parallel private(i) reduction(+:rslt) default(shared)
    {
        int idx = omp_get_thread_num();

        ptr[idx] = (uintptr_t *)malloc( (2*max_size+stride)*sizeof(uintptr_t) );
        if( !ptr[idx] ){
            fprintf(stderr, "Error: cannot allocate space for experiment.\n");
            //#pragma omp critical
            {
                allocErr = -1;
            }
        }else{
            // align v to the stride.
            v[idx] = (uintptr_t *)(stride_in_bytes*(((uintptr_t)ptr[idx]+stride_in_bytes)/stride_in_bytes));

            // touch every page at least a few times
            for(i=0; i<2*max_size; i+=512){
                rslt += v[idx][i];
            }
        }
    }
    if(allocErr != 0)
    {
        goto error;
    }

    // Make a cold run
    out = probeBufferSize(16*stride, stride, pages_per_block, pattern, v, &rslt, latency_only, mode, ONT);
    if(out.status != 0)
        goto error;

    // Run the actual experiment
    if( (NULL==hw_desc) || (hw_desc->cache_levels<=0) ){
        cnt = 0;
        // If we don't know the cache sizes, space the measurements between two default values.
        for(active_buf_len=min_size; active_buf_len<max_size; active_buf_len*=2){
            out = probeBufferSize(active_buf_len, stride, pages_per_block, pattern, v, &rslt, latency_only, mode, ONT);
            if(out.status != 0)
                goto error;
            for(k = 0; k < ONT; ++k) {
                rslts[cnt][k] = out.dt[k];
                counter[cnt][k] = out.counter[k];
            }
            values[cnt++] = ONT*sizeof(uintptr_t)*active_buf_len;

            out = probeBufferSize((int)((double)active_buf_len*1.25), stride, pages_per_block, pattern, v, &rslt, latency_only, mode, ONT);
            if(out.status != 0)
                goto error;
            for(k = 0; k < ONT; ++k) {
                rslts[cnt][k] = out.dt[k];
                counter[cnt][k] = out.counter[k];
            }
            values[cnt++] = ONT*sizeof(uintptr_t)*((int)((double)active_buf_len*1.25));

            out = probeBufferSize((int)((double)active_buf_len*1.5), stride, pages_per_block, pattern, v, &rslt, latency_only, mode, ONT);
            if(out.status != 0)
                goto error;
            for(k = 0; k < ONT; ++k) {
                rslts[cnt][k] = out.dt[k];
                counter[cnt][k] = out.counter[k];
            }
            values[cnt++] = ONT*sizeof(uintptr_t)*((int)((double)active_buf_len*1.5));

            out = probeBufferSize((int)((double)active_buf_len*1.75), stride, pages_per_block, pattern, v, &rslt, latency_only, mode, ONT);
            if(out.status != 0)
                goto error;
            for(k = 0; k < ONT; ++k) {
                rslts[cnt][k] = out.dt[k];
                counter[cnt][k] = out.counter[k];
            }
            values[cnt++] = ONT*sizeof(uintptr_t)*((int)((double)active_buf_len*1.75));
        }
    }else{
        double f;
        int numCaches = hw_desc->cache_levels;
        int numHier   = numCaches+1;
        int llc_idx   = numCaches-1;
        int len = 0, ptsToNextCache, currCacheSize, nextCacheSize, tmpIdx = 0;
        long *bufSizes;

        // Calculate the length of the array of buffer sizes.
        for(j=0; j<numHier; ++j) {
            len += hw_desc->pts_per_reg[j] + 1;
        }
        len++; // To include endpoint.

        // Allocate space for the array of buffer sizes.
        if( NULL == (bufSizes = (long *)calloc(len, sizeof(long))) )
            goto error;

        // Define buffer sizes.
        tmpIdx = 0;
        for(j=0; j<numHier; ++j) {

            ptsToNextCache = hw_desc->pts_per_reg[j]+1;

            /* The lower bound of the first cache region is set to the size, L1/8, as a design decision.
             * All other lower bounds are set to the size of the caches, as observed per core.
             */
            if( 0 == j ) {
                bufSizes[tmpIdx] = hw_desc->dcache_size[0]/(8.0*hw_desc->split[0]);
            } else {
                bufSizes[tmpIdx] = hw_desc->dcache_size[j-1]/hw_desc->split[j-1];
            }
            currCacheSize = bufSizes[tmpIdx];

            /* The upper bound of the final "cache" region (memory in this case) is set to 12 times the
             * size of the LLC so that all threads cumulatively will exceed the LLC by a factor of 12.
             * All other upper bounds are set to the capacity of the cache, as observed per core.
             */
            if( llc_idx+1 == j ) {
                nextCacheSize = 12LL*(hw_desc->dcache_size[llc_idx])/hw_desc->split[llc_idx];
                bufSizes[tmpIdx+ptsToNextCache] = nextCacheSize;
            } else {
                nextCacheSize = hw_desc->dcache_size[j]/hw_desc->split[j];
            }

            /* Choose a factor "f" to grow the buffer size by, such that we collect the user-specified
             * number of samples between each cache size, evenly distributed in a geometric fashion
             * (i.e., sizes will be equally spaced in a log graph).
             */
            for(k = 1; k < ptsToNextCache; ++k) {
                f = pow(((double)nextCacheSize)/currCacheSize, ((double)k)/ptsToNextCache);
                bufSizes[tmpIdx+k] = f*currCacheSize;
            }

            tmpIdx += ptsToNextCache;
        }

        cnt=0;
        for(j=0; j<len; j++){
            active_buf_len = ((long)bufSizes[j])/sizeof(uintptr_t);
            out = probeBufferSize(active_buf_len, stride, pages_per_block, pattern, v, &rslt, latency_only, mode, ONT);
            if(out.status != 0)
                goto error;
            for(k = 0; k < ONT; ++k) {
                rslts[cnt][k] = out.dt[k];
                counter[cnt][k] = out.counter[k];
            }
            values[cnt++] = bufSizes[j];
        }

        free(bufSizes);
    }

    // Free each thread's memory.
    for(j=0; j<ONT; ++j){
        free(ptr[j]);
    }
    return 0;

error:
    // Free each thread's memory.
    for(j=0; j<ONT; ++j){
        free(ptr[j]);
    }
    return -1;
}

int get_thread_count() {

    int threadNum = 1;

    //#pragma omp parallel default(shared)
    {
        if(!omp_get_thread_num()) {
            threadNum = omp_get_num_threads();
        }
    }

    return threadNum;
}

void print_core_affinities(FILE *ofp) {

    int k, ONT;
    int *pinnings = NULL;

    // Get the number of threads.
    ONT = get_thread_count();

    // List of core affinities in which the index is the thread ID.
    pinnings = (int *)malloc(ONT*sizeof(int));
    if( NULL == pinnings ) {
        fprintf(stderr, "Error: cannot allocate space for experiment.\n");
        return;
    }

    //#pragma omp parallel default(shared)
    {
        int idx = omp_get_thread_num();

        pinnings[idx] = sched_getcpu();
    }

    fprintf(ofp, "# Core:");
    for(k=0; k<ONT; ++k) {
        fprintf(ofp, " %d", pinnings[k]);
    }
    fprintf(ofp, "\n");

    free(pinnings);

    return;
}
