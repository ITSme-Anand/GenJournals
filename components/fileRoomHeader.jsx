import { View, Text } from 'react-native'
import React from 'react'
import { Stack } from 'expo-router'
import { TouchableOpacity } from 'react-native'
import {icons} from '../constants'
import { Image } from 'react-native'
import { Entypo } from '@expo/vector-icons'
import { widthPercentageToDP as wp , heightPercentageToDP as hp } from 'react-native-responsive-screen'

const fileRoomHeader = ({fileDetails , router , handlePressBack}) => {
  return (
    <View className="flex-row items-center gap-4 my-2 border-b border-gray-600">
                        <TouchableOpacity onPress={()=>{
                            handlePressBack(fileDetails)
                            router.back()
                        }}>
                            <Entypo name="chevron-left" size={hp(4)} color="#737373" />
                            

                        </TouchableOpacity>
    </View>
  )
}

export default fileRoomHeader