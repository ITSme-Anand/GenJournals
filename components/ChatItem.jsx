import { View, Text } from 'react-native'
import React from 'react'
import { Card } from 'react-native-paper'
import { icons } from '../constants'
import { Avatar } from 'react-native-paper'
import {IconButton} from 'react-native-paper'
const ChatItem = ({item, noborder , router}) => {
  const LeftContent = props => <Avatar.Icon {...props} icon={icons.chatBot} />
  const openChat = ()=>{
    router.push({pathname:'/chatPage', params:item});
  }
  return (
    

    <Card
        className={`justify-between w-[100vw] pr-2 mb-1 pb-2 ${noborder?'':'border-b  border-b-neutral-200'}`}
        mode='elevated'
        onPress={openChat}  
        style={{borderRadius:0 , backgroundColor:"#1e1e2e"}}  
        
        
        
    >
      
      <View className="gap-1">
                <View className="flex-row justify-between">
                    <Card.Title left={LeftContent} className=" w-[20%] " />
                    <View className="flex-row justify-between  w-[80%] items-center ">
                        
                        <Text className="font-semibold text-xl  text-white" > {item['Name']} </Text>
                        <Card.Actions>
                          <IconButton
                                
                                icon={icons.pencil}
                                iconColor='#FFFFFF'
                                size={20}
                                onPress={()=>{console.log('edit button is pressed')}}
                                mode='outlined'
                                
                          />
                        </Card.Actions> 

                    </View>
                    
                    
                </View>
                
                <Text className="font-medium text-neutral-500 px-3"> {item['Description']}</Text>
      </View>
                
                
    </Card>
    
    
  )
} 

export default ChatItem