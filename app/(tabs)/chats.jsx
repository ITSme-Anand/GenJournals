import { View, Text, SafeAreaView } from 'react-native'
import React from 'react'
import ChatList from '../../components/chatList'

const Chat = () => {
  
  const chatItems = [{'id':1,'Name':'AI Friend','TableName':'AIfriend','Time':'21:00','Last Message':null,'Description':'An AI friend who is always there for you'},
                     {'id':2,'Name':'Future-Self','TableName':'futureSelf','Time':'21:00','Last Message':null,'Description':"I'm You, From the future!"},
  ];
  
  return (
    <SafeAreaView className="h-screen">
      <View className="flex-1 bg-white">
        {
          chatItems.length>0?(<ChatList users={chatItems} />):(
            <View className="flex items-center" >
                <Text>Hello </Text>
            </View>
          )
        }
      </View>
    </SafeAreaView>
  )
}

export default Chat