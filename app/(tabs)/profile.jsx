import { View, Text } from 'react-native'
import React from 'react'
import { SafeAreaView } from 'react-native'
import { Card } from 'react-native-paper'
import {IconButton} from 'react-native-paper'
import { icons } from '../../constants'
import { useState } from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { ScrollView } from 'react-native'
import {TextInput} from 'react-native-paper'
import { Button } from 'react-native-paper'
import { useRef } from 'react'
import { onAuthStateChanged, signOut } from 'firebase/auth'
import auth from '../../services/firebaseAuth'

const profile = () => {
  const [userName, setUserName] = useState(null)
  const [editUserName, seteditUserName] = useState(false)
  const [NewUserName, setNewUserName] = useState(null)
  const textRef = useRef(null)
  AsyncStorage.getItem('userName').then(
    (cred)=>{
      setUserName(cred);
      textRef.current = cred;
      console.log('textRef is ', textRef.current)
    }
  )
  
  const UserNameEditor = ()=>{
    seteditUserName(true)
    
  }
  const handleUserNameSave = ()=>{
    seteditUserName(false)
    console.log('new username is ', NewUserName)
    
    AsyncStorage.setItem('userName',NewUserName).then( ()=>{
      console.log('userName set')
      //setUserName(NewUserName)
    }
    )
    
  }
  const handleLogOut = ()=>{
    //signOut(auth)
    onAuthStateChanged(auth , (user)=>{
      if(user){
        console.log('user still logged in! uncomment the logout line!')
      }
      else{
        console.log('user is logged out')
      }
    })
  }
  return (
    <SafeAreaView className="bg-primary h-[100vh]">
    <ScrollView>
    <View className="mt-20" >
      <View className="flex flex-column items-center w-[95vw] self-center">
        <Card
          className=" bg-primary text-center border-none flex-column"  
          style={{borderRadius:0}}
                          
        >
          <View className=" flex flex-row  items-center">
            <Text className="text-white text-2xl text-center font-bold"> UserName </Text>
              <Card.Actions>
                        <IconButton
                              
                              icon={icons.pencil}
                              iconColor='#FFFFFF'
                              size={20}
                              onPress={UserNameEditor}
                              mode='outlined'
                              className="bg-black-100"
                        />
              </Card.Actions> 
          </View>
          <TextInput 
              value={(!editUserName)?userName:textRef.current} 
              ref={textRef}
              onChangeText={value=>{
                textRef.current = value
                setNewUserName(value)
              }}
              mode='outlined'
              className="bg-primary w-[90vw] self-center"
              textColor='white'
              style={{borderRadius:10, borderTopStartRadius:10 , borderTopRightRadius:10}}
              editable={editUserName?true:false}
              autoFocus={true}
              
              
          />
          {editUserName?(
           <Card.Actions>
              <View className="flex flex-row justify-between w-[90vw] self-center">
                  <Button 
                  className=" text-white bg-black-200 w-[49%]"
                  onPress={()=>{seteditUserName(false);textRef.current=userName;}}
                  mode='contained'
                  style={{borderRadius:0, borderColor:'#737373'}}
                
                  >
                    Cancel
                  </Button>
                  <Button
                    onPress={handleUserNameSave}
                    className=" bg-black-200 text-white w-[49%]"
                    mode='contained'
                    style={{borderRadius:0 , borderColor:'#737373'}}
                  >
                    Save
                  </Button>
              </View>
            </Card.Actions>

          ):(<></>)}
          
                  
        </Card>
       </View>
       <View>
        <Button
        className="bg-red-500 text-white text-lg w-[90vw] self-center mt-10"
        style={{borderRadius:6}}
        textColor='white'
        onPress={handleLogOut}
        >
            Logout
        </Button>
       </View>
    </View>
    </ScrollView>
    </SafeAreaView>
  )
}

export default profile