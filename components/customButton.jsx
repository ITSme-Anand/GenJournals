import { View, Text } from 'react-native'
import React from 'react'
import { TouchableOpacity } from 'react-native-gesture-handler'


const customButton = ({title, handlePress, containerStyles, textStyles, isLoading}) => {
  return (
    <TouchableOpacity 
      onPress={handlePress}
      className={`bg-secondary rounded-xl min-h-[60px] justify-center items-center ${containerStyles} ${isLoading? 'opactiy-50' : ''}`}
      activeOpacity={0.7}
      disabled={isLoading}
    >
        <Text className={`text-primary font-psemibold text-lg px-2 ${textStyles}`}> {title }</Text> 
    </TouchableOpacity>    
  )
}

export default customButton