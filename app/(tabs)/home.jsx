import { View, Text } from 'react-native'
import React from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { useState } from 'react'
import { SafeAreaView } from 'react-native'
import { GestureHandlerRootView } from 'react-native-gesture-handler'
import { ScrollView } from 'react-native'
import { images } from '../../constants'
import { Card  } from 'react-native-paper'
import {IconButton} from 'react-native-paper'
import {icons} from '../../constants'
import { Image } from 'react-native'


const home = () => {
  const [userName, setUserName] = useState(null)
  AsyncStorage.getItem('userName').then(
    (cred)=>{
      setUserName(cred);
    }
  )
  return (
    <SafeAreaView className="bg-primary h-screen">
      
        <ScrollView>
          <View className="bg-primary flex flex-col items-center">
            <View className="flex flex-col items-center">
              <Image source={images.logo}
                        resizeMode='contain'
                        className=' border border-white w-[75vw] h-[25vh]'
                        

              />
            </View>
            <View>
            <Card
                className="w-[95vw] bg-primary text-center"  
                         
              >
                {userName && (<Text className="text-center text-3xl" style={{color:'white'}}>Welcome {`${userName}`} </Text>)} 
                
              </Card>
            </View>
            <View className="text-center flex flex-col items-center">
              <Card
                className="w-[95vw] bg-black-100 pt-4 my-3 border border-gray-700 rounded-xl text-center"
                mode='elevated'  
                         
              >
                <View className="flex flex-col items-center">
                <Text className="text-white text-2xl mb-2 text-center font-bold" > Message Of the Day </Text>
                <View  className="border-t w-[70vw] border-white" />

                <Text className="text-white text-lg mb-1 mt-3 text-center" style={{fontStyle:'italic'}}> Fate Whispers to the Warrior,{'\n'} "You cannot WithStand the Storm!",{'\n'} And the Warrior Whispers back {'\n'} "I'm the Storm"  </Text>
                </View>
                <Card.Actions>
                    <IconButton
                          
                          icon={icons.pencil}
                          iconColor='#FFFFFF'
                          size={20}
                          onPress={()=>{console.log('edit button is pressed')}}
                          mode='outlined'
                          className="bg-black-100"
                    />
                </Card.Actions> 
              </Card>
            </View>
            

          </View>
        </ScrollView>
      
    </SafeAreaView>
        
  )
}

export default home