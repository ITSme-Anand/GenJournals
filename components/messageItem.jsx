import { View, Text } from 'react-native'
import React from 'react'

import { widthPercentageToDP as wp, heightPercentageToDP as hp } from 'react-native-responsive-screen'
const messageItem = ({message,}) => {
    //console.log('inside message item', message)
    if(message['sender']==='User'){
        return (
            <View className="flex-row justify-end mb-3 mr-3">
                <View style={{width: wp(80)}}>
                    <View className="flex self-end p-3 rounded-2xl bg-white border border-neutral-200">
                        <Text style={{fontSize: hp(1.9)}}>
                            {message['message']}
                        </Text>
                    </View>
                </View>
            </View>
        )
    }
    return(
        <View style={{width:wp(80)}} className="flex-row justify-start mb-3 ml-3">
                
                    <View className="flex self-start p-3 px-4 rounded-2xl bg-indigo-100 border border-indigo-200">
                        <Text style={{fontSize: hp(1.9)}}>
                            {message['message']}
                        </Text>
                    </View>
                
        </View>
    )
    
}

export default messageItem