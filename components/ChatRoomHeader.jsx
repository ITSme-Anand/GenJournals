import { View, Text } from 'react-native'
import React from 'react'
import { Stack } from 'expo-router'
import { TouchableOpacity } from 'react-native'
import {icons} from '../constants'
import { Image } from 'react-native'
import { Entypo } from '@expo/vector-icons'
import { widthPercentageToDP as wp , heightPercentageToDP as hp } from 'react-native-responsive-screen'
const ChatRoomHeader = ({user, router}) => {
  return (
    <View className="flex-row items-center gap-4">
                        <TouchableOpacity onPress={()=>router.back()}>
                            <Entypo name="chevron-left" size={hp(4)} color="#737373" />
                            

                        </TouchableOpacity>
                        <View>
                            <Text style={{fontSize: hp(2.5)}} className="text-neutral-700 font-medium">
                                {user?.Name}
                            </Text>
                        </View>
    </View>
  )
}

export default ChatRoomHeader