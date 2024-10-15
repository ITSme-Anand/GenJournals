import { View, Text, Platform } from 'react-native'
import React from 'react'
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Card } from 'react-native-paper';

const ios = Platform.OS=='ios';
const customHeader = () => {

    const {top} = useSafeAreaInsets();

  return (
    <View style={{paddingTop:ios?top:top+10}} className="flex-column items-center w-[100vw] py-4 bg-primary shadow"> 
      
      <Card
                className="w-[100vw] bg-black-100 text-center py-2"
                style={{borderRadius:0}}
                         
              >
                <Text className="text-center text-2xl text-secondary-200" > Chats </Text> 
                
                
              </Card>
    </View>
  )
}

export default customHeader