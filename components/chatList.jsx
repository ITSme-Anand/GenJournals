import { View, Text } from 'react-native'
import React from 'react'
import { FlatList } from 'react-native'
import ChatItem from './ChatItem'
import { useRouter } from 'expo-router'
const ChatList = ({users}) => {

    const Router = useRouter();
  return (
    
    <View className="flex-1 bg-primary">
      <FlatList
      data={users} 
      contentContainerStyle={{flex:1,paddingVertical:25}}
      keyExtractor={item=> item['id']}
      showsVerticalScrollIndicator = {false}
      renderItem={({item , index})=><ChatItem noborder={users.length==index+1} router={Router} item={item} index={index} />} 
      
      
      />
        
      
    </View>
  )
}

export default ChatList