/*! \file CANDORObject.h*/
//
// Created by tony on 19/03/24.
//

#ifndef CANDOR_INCLUDE_CANDOROBJECT_H_
#define CANDOR_INCLUDE_CANDOROBJECT_H_
#include <string>
#include <vector>
#include <memory>
namespace CANDOR {
/**
 * @ingroup  CANDOR_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class CANDORObject CANDOR/RAMIAObject.h
 * @brief A generic object class to link string or void * pointers
 * @todo to finish the functions of setting void * pointers
 */
class CANDORObject {
 public:
  CANDORObject() {}
  ~CANDORObject() {}
  std::string objStr;
  void *objPointer = nullptr;
  int64_t objSize = 0;
  int64_t objId = -1;
  /**
   * @brief to set the string
   * @param str the string
   * @return void
   */
  void setStr(std::string str);
  /**
   * @brief to get the string
   * @return the objStr
   */
  std::string getStr();
};

/**
 * @ingroup  CANDOR_lib_bottom
 * @typedef CANDORObjectPtr
 * @brief The class to describe a shared pointer to @ref  CANDORObject

 */
typedef std::shared_ptr<class CANDOR::CANDORObject> CANDORObjectPtr;
/**
 * @ingroup  CANDOR_lib_bottom
 * @def newAbstractIndex
 * @brief (Macro) To creat a new @ref  CANDORObject shared pointer.
 */
#define newCANDORObject std::make_shared<CANDOR::CANDORObject>
/**
 * @}
 */
} // CANDOR

#endif //CANDOR_INCLUDE_CANDOROBJECT_H_
