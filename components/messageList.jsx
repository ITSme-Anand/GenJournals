import { View, Text } from 'react-native'
import React from 'react'
import { ScrollView } from 'react-native'
import MessageItem from './messageItem'

const MessageList = ({messages,}) => {
  return (
    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingTop: 10}} inverted={true} >
      {
        messages.map((message, index)=>{
          return (
            <MessageItem message={message} key={index}/>
          )
        })
      }
    </ScrollView>
  )
}

export default MessageList