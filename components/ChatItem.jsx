import { View, Text } from 'react-native'
import React from 'react'
import { Card } from 'react-native-paper'
import { icons } from '../constants'
import { Avatar } from 'react-native-paper'
import {IconButton} from 'react-native-paper'
import { useState } from 'react'
import { useRef } from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'
import {TextInput} from 'react-native-paper'
import { Button } from 'react-native-paper'
const ChatItem = ({item, noborder , router}) => {
  const LeftContent = props => <Avatar.Icon {...props} icon={icons.chatBot} />
  const openChat = ()=>{
    router.push({pathname:'/chatPage', params:item});
  }
  const [chatName, setchatName] = useState(null)
  const [editchatName, seteditchatName] = useState(false)
  const [NewchatName, setNewchatName] = useState(null)
  const textRef = useRef(null)
  
  AsyncStorage.getItem(item['id']==1?'AIfriendName':'futureSelfName').then(
    (cred)=>{
      setchatName(cred);
      textRef.current = cred;
      console.log('textRef is ', textRef.current)
    }
  )
  const chatNameEditor = ()=>{
    seteditchatName(true)
    
  }
  const handlechatNameSave = ()=>{
    seteditchatName(false)
    console.log('new chatName is ', NewchatName)
    
    AsyncStorage.setItem(item['id']==1?'AIfriendName':'futureSelfName',NewchatName).then( ()=>{
      console.log('userName set')
      setchatName(NewchatName)
    } 
    )
    
  }
  return (
    

    <Card
        className={`justify-between w-[100vw] pr-2 mb-1 pb-2 ${noborder?'':'border-b  border-b-neutral-200'}`}
        mode='elevated'
        onPress={openChat}  
        style={{borderRadius:0 , backgroundColor:"#1e1e2e"}}  
        
        
        
    >
      
      <View className="w-[100vw] bg-primary">
        <View className="flex flex-col items-center self-center bg-primary ">
                <View className="flex-row justify-between">
                    <Card.Title left={LeftContent} className=" w-[20%] " />
                    <View className="flex-row justify-between  w-[80%] items-center ">
                        
                        <TextInput 
                          value={(!editchatName)?chatName:textRef.current} 
                          ref={textRef}
                          onChangeText={value=>{
                            textRef.current = value
                            setNewchatName(value)
                          }}
                          mode={editchatName?'outlined':'flat'}
                          className="bg-primary w-[60vw] self-center font-semibold text-xl  text-white"
                          textColor='white'
                          style={editchatName?{borderRadius:10, borderTopStartRadius:10 , borderTopRightRadius:10}:{}}
                          editable={editchatName?true:false}
                          autoFocus={true}
                          underlineColor='#161622'
                          
                          
                        />
                        
                        <Card.Actions>
                          <IconButton
                                
                                icon={icons.pencil}
                                iconColor='#FFFFFF'
                                size={20}
                                onPress={chatNameEditor}
                                mode='outlined'
                                
                          />
                        </Card.Actions> 

                    </View>

                    
                    
                </View>
              {editchatName?(
                      <Card.Actions>
                          <View className="flex flex-row justify-between w-[90vw] self-center">
                              <Button 
                              className=" text-white bg-black-200 w-[49%]"
                              onPress={()=>{seteditchatName(false);textRef.current=chatName;}}
                              mode='contained'
                              style={{borderRadius:0, borderColor:'#737373'}}
                            
                              >
                                Cancel
                              </Button>
                              <Button
                                onPress={handlechatNameSave}
                                className=" bg-black-200 text-white w-[49%]"
                                mode='contained'
                                style={{borderRadius:0 , borderColor:'#737373'}}
                              >
                                Save
                              </Button>
                          </View>
                        </Card.Actions>

                      ):(<></>)}
                
                <Text className="font-medium text-neutral-500 px-3 self-start"> {item['Description']}</Text>
                
      </View>
    </View>  
                
    </Card>
    
    
  )
} 

export default ChatItem