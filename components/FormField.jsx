import { View, Text } from 'react-native'
import React from 'react'
import { TextInput } from 'react-native'
import { useState } from 'react'
import { TouchableOpacity } from 'react-native-gesture-handler'
import {icons} from "../constants"
import { Image } from 'react-native'
const FormField = ({title , value , placeholder ,handleChangeText,otherStyles, ...props}) => {
  const [showPassword, setShowPassword] = useState(false)
  return (
    <View className={`space-y-2 ${otherStyles}`}>
      <Text className=" font-pmedium text-xs text-gray-100">
        {title}
      </Text>
      <View className="w-[90vw] h-16 bg-black-100 border-2 border-black-200 rounded-2xl focus:border-secondary flex flex-row items-center px-4">
        <TextInput 
            className="flex-1 text-white font-psemibold text-base "
            value={value}
            placeholder={placeholder}
            placeholderTextColor={'#7b7b8b'}
            onChangeText={handleChangeText}
            secureTextEntry={title==='Password' && !showPassword}
        />
        {title==="Password" &&(
          <TouchableOpacity onPress={()=>
            setShowPassword(!showPassword)
          }>
            <Image source={icons.eye} 
            className="w-6 h-6"
            resizeMode='contain'
            
            />
            
          </TouchableOpacity>
        ) }
      </View>

    </View>
  )
}

export default FormField