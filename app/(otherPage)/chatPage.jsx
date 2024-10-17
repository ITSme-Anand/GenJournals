import { View, Text, Alert } from 'react-native'
import React, { useRef } from 'react'
import { useLocalSearchParams, useRouter } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import ChatRoomHeader from '../../components/ChatRoomHeader'
import { SafeAreaView } from 'react-native'
import { GestureHandlerRootView } from 'react-native-gesture-handler'
import { ScrollView } from 'react-native'
import MessageList from '../../components/messageList'
import { useState } from 'react'
import { widthPercentageToDP as wp, heightPercentageToDP as hp } from 'react-native-responsive-screen'
import { TextInput } from 'react-native'
import {TouchableOpacity} from 'react-native-gesture-handler'
import { Feather } from '@expo/vector-icons'
import {icons} from "../../constants"
import { Image } from 'react-native';
import CustomKeyboardView from '../../components/CustomKeyboardView';
import * as SQLite from 'expo-sqlite';
import { useEffect } from 'react'
import loadDatabase from '../../services/databaseIn'
import { Suspense } from 'react'
import { SQLiteProvider } from 'expo-sqlite/next'
import { Platform } from 'react-native'
import { KeyboardAvoidingView } from 'react-native'
import axios from 'axios'
import { Keyboard } from 'react-native'
const ios = Platform.OS =="ios"

const chatPage = () => { 
    
    const item = useLocalSearchParams(item);
    console.log('item is ', item);
    const router = useRouter();
    const [messages, setMessages] = useState(null)
    const textRef = useRef('');
    const inputRef = useRef(null);
    const [marginb, setmarginb] = useState(0.5)
    const [keyboardOn, setKeyboardOn] = useState(0)
    const [dbLoaded, setDbLoaded] = useState(false)
    const db = SQLite.useSQLiteContext();

    //this is useEffect for keyBoard
    useEffect(() => {
      const keyboardDidShowListener = Keyboard.addListener('keyboardDidShow', _keyboardDidShow);
      const keyboardDidHideListener = Keyboard.addListener('keyboardDidHide', _keyboardDidHide);
  
      // Cleanup listeners on unmount
      return () => {
        keyboardDidShowListener.remove();
        keyboardDidHideListener.remove();
      };
    }, []);
  
    const _keyboardDidShow = () => {
      // Handle keyboard shown
      setKeyboardOn(20)
      setmarginb(2.7)
    };
  
    const _keyboardDidHide = () => {
      // Handle keyboard hidden
      setKeyboardOn(5)
      setmarginb(0.5)
    };

  

    /* async function createTables(){
      try{
        //console.log('inside createtables function')
        //DROP TABLE IF EXISTS FutureSelf;
        
        await db.execAsync(
          ` 
            
            
            CREATE TABLE IF NOT EXISTS futureSelf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL CHECK (sender IN ('User', 'ChatBot')),
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS AIfriend (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL CHECK (sender IN ('User', 'ChatBot')),
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            
          `
        ) 
      }
      catch(err){
          console.error(err)
      }
    }  
 */
    useEffect(()=>{
      db.withTransactionAsync(async ()=>{
        await fetchMessages();
        
      })
    } , [db])

    async function fetchMessages(){
      try{
        const result = await db.getAllAsync(
          `SELECT * FROM ${item['TableName']}
          `) 
        
        setMessages(result)
      }
      catch(e){
        console.log('error here in fetch messages')
        console.error(e)
      }
    }

  

    const postMessageToFlask = async (message)=>{
        let response = await axios.post('http://192.168.0.102:5000/sendMessage', { userPrompt: message },{ headers: { 'Content-Type': 'application/json' } } );
        const Response = response.data["response"]

        console.log('Parsed server response:', Response);
        try{
          await BotSendMessage(Response)
        }
        catch(err){
          console.log('error happening here da')
          console.error(err)
        }
        
        
    }

    
    const UserSendMessage = ()=>{
      setKeyboardOn(0)
      setmarginb(0.5)
      handlesendMessage('User');
      
    }

    const BotSendMessage = async (response) =>{
      console.log('inside botSendMessage function')
      console.log('the received response is ', response)
      try{
        await db.runAsync(
          `
            INSERT INTO ${item['TableName']} (sender , message) VALUES (?, ?);
          `,
          [
            'ChatBot',
            response
          ]
      );
      }catch(err){
        console.error(err)
      }
      await fetchMessages();
      
    }

    const handlesendMessage = async (sender)=>{
      //console.log('inside handlesendMessage function. the sender is', sender)
      let message = textRef.current.trim();
      if(!message) return;
      try{
        //console.log(message)
        db.withTransactionAsync(
          async ()=>{

            try{
              await db.runAsync(
                `
                  INSERT INTO ${item['TableName']} (sender , message) VALUES ('${sender}', '${message}');
                  
                `
            );
            }
            catch(err){
              console.log('error is happening inside the execAsync line 194')
              console.error(err)
            }
            
          }
          
        )
        try{
          await fetchMessages();
          
        }
        catch{
          console.log('error is happening while doing the fetch messages')
        }
        try{
          await postMessageToFlask(message);
          
        }
        catch{
          console.log('error is happening while doing the postMessage to flask')
        }
        
        console.log('transaction completed and the message sent to the flask successfully!')
        textRef.current = "";
        if(inputRef) inputRef?.current?.clear();

      }
      catch(err){
        Alert.alert('error has occured');
        console.log(err)
      }
    }

    

  return (
    
    <SafeAreaView className="bg-primary h-full">

    <GestureHandlerRootView>
        
        <KeyboardAvoidingView
            behavior={ios?'padding':'height'}
            style={{flex:1}}
            keyboardVerticalOffset={keyboardOn}
            
            

        >
          <ScrollView 
            style={{flex:1}}
            bounces={false}
            showsVerticalScrollIndicator={false}
            contentContainerStyle= {{flex:1}}
            inverted={true}
        >
          <View className="flex-1 mt-6 h-screen w-full">
            <StatusBar style="dark" />
            <ChatRoomHeader user={item} router={router} /> 
            <View className="h-3 border-b border-neutral-300"/>
            <View className="flex-1 justify-around bg-primary overflow-visible h-[90vh]">
              <View className="flex-1">
                <MessageList messages={messages?messages:['loading']}/>
              </View>
              <View style={{marginBottom: hp(marginb)}} className="pt-2">
                    <View className="flex-row justify-between bg-secondary border p-2 mb-2 border-neutral-300 rounded-full pl-5 mx-3">
                      <TextInput 
                        ref={inputRef}
                        onChangeText={value=> textRef.current = value}
                        placeholder="Type message..."
                        className="flex-1 mr-2"
                        style={{fontSize: hp(2)}}
                        
                        
                      />
                      <TouchableOpacity onPress={UserSendMessage} className="bg-neutral-200 p-1 mr-[1px] rounded-full">
                          <Feather name='send' size={hp(2.7)} color="#737373" />

                      </TouchableOpacity>
                    </View>
              </View>
            </View>
          </View>
          </ScrollView>
      </KeyboardAvoidingView>
       
    </GestureHandlerRootView>
</SafeAreaView>
  )
}

export default chatPage