import { View, Text, Image } from 'react-native'
import FontAwesome from '@expo/vector-icons/FontAwesome';
import React from 'react'
import { Tabs,Redirect } from 'expo-router'
import {icons} from "../../constants"
import { FA5Style } from '@expo/vector-icons/build/FontAwesome5'
import customHeader from '../../components/customHeader';
import { SQLiteProvider } from 'expo-sqlite/next';

const TabIcon = ({icon , color , name , focused}) =>{
  return (
    <View className="items-center justify-center gap-2">
      <Image 
        source={icon}
        resizeMode='contain'
        tintColor={color}
        className="w-6 h-6"
      />
      <Text className={`${focused? 'font-psemibold': 'font-pregular'} text-xs`} style={{color:color}}>
        {name}
      </Text>
    </View>
  )
}

const tabsLayout = () => {
  return (
    <SQLiteProvider databaseName='chatBot.db' useSuspense>
    <Tabs screenOptions={{ tabBarActiveTintColor: '#FFA001' ,tabBarShowLabel:false , tabBarInactiveTintColor:'#CDCDE0' , tabBarStyle:{
      backgroundColor: '#161622',
      borderTopWidth: 1,
      borderTopColor: '#232533',
      height:84,
    } }}>
      <Tabs.Screen
        name="home"
        options={{
          title: 'Home',
          tabBarIcon: ({ color , focused }) =>(<TabIcon color={color} focused={focused} icon={icons.home} name="home" />),
          headerShown:false
        }}
      />
      
        <Tabs.Screen
          name="fileSpace"
          options={{
            title: 'Files',
            tabBarIcon: ({ color , focused}) => (<TabIcon color={color} focused={focused} icon={icons.folder} name="Files" />),
            headerShown:false,
          }}
        />
      
      
      <Tabs.Screen
        name="chats"
        options={{
          title: 'Chat',
          tabBarIcon: ({ color , focused}) => (<TabIcon color={color} focused={focused} icon={icons.chatBot} name="Chat" />),
          header: customHeader,
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'profile',
          tabBarIcon: ({ color , focused}) => (<TabIcon color={color} focused={focused} icon={icons.profile} name="profile" />),
          
        }}
      />

      
      
    </Tabs>
    </SQLiteProvider>
  );
}

export default tabsLayout