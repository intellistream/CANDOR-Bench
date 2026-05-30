package internal

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const defaultDiffPlotScript = "../utils/scripts/plot_recall_diff.py"
const pythonEnvVar = "PYTHON_BIN"

func resolveScriptPath(explicit, defaultRel string) string {
	explicit = strings.TrimSpace(explicit)
	if explicit != "" {
		return explicit
	}

	candidates := make([]string, 0, 3)
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		candidates = append(candidates,
			filepath.Join(exeDir, defaultRel),
			filepath.Join(exeDir, "..", defaultRel),
		)
	}
	candidates = append(candidates, defaultRel)

	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	if len(candidates) > 0 {
		return candidates[0]
	}
	return defaultRel
}

func resolvePythonInterpreter() (string, error) {
	if env := strings.TrimSpace(os.Getenv(pythonEnvVar)); env != "" {
		if _, err := exec.LookPath(env); err == nil {
			return env, nil
		}
	}
	candidates := []string{"python3", "python"}
	for _, name := range candidates {
		if _, err := exec.LookPath(name); err == nil {
			return name, nil
		}
	}
	return "", fmt.Errorf("no python interpreter found (tried %v and env %s)", candidates, pythonEnvVar)
}

func resolveDiffPlotScript() string {
	return resolveScriptPath("", defaultDiffPlotScript)
}

func figurePath(csvPath string) (string, error) {
	baseDir := filepath.Dir(csvPath)
	parentDir := filepath.Dir(baseDir)
	targetDir := baseDir
	if parentDir != "" && parentDir != "." && parentDir != string(filepath.Separator) {
		targetDir = parentDir
	}
	figDir := filepath.Join(targetDir, "figures")
	if err := os.MkdirAll(figDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create figures directory %s: %w", figDir, err)
	}
	base := strings.TrimSuffix(filepath.Base(csvPath), filepath.Ext(csvPath))
	return filepath.Join(figDir, base+".png"), nil
}

func runDiffPlot(config *Config, diffCSVPath, mode string) {
	script := resolveDiffPlotScript()
	if script == "" || diffCSVPath == "" {
		return
	}

	if _, err := os.Stat(diffCSVPath); err != nil {
		log.Printf("Query mode %s: diff CSV %s not accessible (%v); skipping plot script", mode, diffCSVPath, err)
		return
	}

	if _, err := os.Stat(script); err != nil {
		if os.IsNotExist(err) {
			log.Printf("Query mode %s: diff plot script %s not found; skipping", mode, script)
		} else {
			log.Printf("Query mode %s: cannot access diff plot script %s: %v", mode, script, err)
		}
		return
	}

	outPath, err := figurePath(diffCSVPath)
	if err != nil {
		log.Printf("Query mode %s: failed to prepare figures directory: %v", mode, err)
		return
	}
	pythonExe, err := resolvePythonInterpreter()
	if err != nil {
		log.Printf("Query mode %s: diff plot skipped (python missing): %v", mode, err)
		return
	}
	outWoFuture := strings.Replace(outPath, ".png", "_wo_future.png", 1)
	cmd := exec.Command(
		pythonExe,
		script,
		"--csv", diffCSVPath,
		"--out", outPath,
		"--out-wo-future", outWoFuture,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Query mode %s: plot script failed: %v\nCommand: %s\nOutput:\n%s", mode, err, strings.Join(cmd.Args, " "), string(output))
		return
	}
	log.Printf("Query mode %s: wrote diff plot to %s", mode, outPath)
	if len(output) > 0 {
		log.Printf("Query mode %s plot script output:\n%s", mode, strings.TrimSpace(string(output)))
	}
}
