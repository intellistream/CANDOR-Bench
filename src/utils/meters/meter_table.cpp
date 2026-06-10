#include <utils/meters/meter_table.h>
#include <utils/meters/esp_meter_uart/esp_meter_uart.hpp>
#include <utils/meters/intel_meter/intel_meter.hpp>

namespace DIVERSE_METER {
/**
 * @note revise me if you need new loader
 */
DIVERSE_METER::MeterTable::MeterTable() {
  meterMap["espUart"] = newEspMeterUart();
  meterMap["intelMsr"] = newIntelMeter();
}

}