//
// Created by tony on 19/03/24.
//

#include <index/candor_object.h>

namespace CANDOR {
void CANDORObject::setStr(std::string str) {
  objStr = str;
  objSize = str.size();
}
std::string CANDORObject::getStr() {
  return objStr;
}

} // CANDOR