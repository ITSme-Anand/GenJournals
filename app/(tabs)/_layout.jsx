import { View, Text, Image } from 'react-native'
import FontAwesome from '@expo/vector-icons/FontAwesome';
import React from 'react'
import { Tabs,Redirect } from 'expo-router'
import {icons} from "../../constants"
import { FA5Style } from '@expo/vector-icons/build/FontAwesome5'
import customHeader from '../../components/customHeader';
import { SQLiteProvider } from 'expo-sqlite/next';
import loadDatabase from '../../services/databaseIn';
import * as SQLite from 'expo-sqlite';

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
  db = SQLite.useSQLiteContext()
  const createTablesAtFirst = async ()=>{
    try{//create a primary id for all the userFile tables
       /* DROP TABLE IF EXISTS userFile1;
      DROP TABLE IF EXISTS userFile2;
      DROP TABLE IF EXISTS tableMetaData;
      INSERT INTO userFile1 (id,entry) VALUES (1,'Gratitude Journal');
      INSERT INTO userFile1 (id,entry) VALUES (2,'');
      INSERT INTO userFile2 (id,entry) VALUES (1,'Goal Trackers');
      INSERT INTO userFile2 (id,entry) VALUES (2,'');
      INSERT INTO tableMetaData (id,tableName,title,line1,line2, line3) VALUES (1,'userFile1','Gratitude Journal','','','');
      INSERT INTO tableMetaData (id,tableName,title,line1,line2, line3) VALUES (2,'userFile2','Goal Trackers','','',''); */
      await db.execAsync(
        `
        CREATE TABLE IF NOT EXISTS userFile1 (
        primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        entry TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS userFile2 (
        primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        entry TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS tableMetaData (
        primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        tableName TEXT NOT NULL,
        title TEXT,
        line1 TEXT,
        line2 TEXT,
        line3 TEXT
        );
        
        `
      )
    }catch(err){
      console.error('error while creating permanent tables')
      console.error(err)
    }
    
  }

  async function createTables(){
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

  createTablesAtFirst()
  createTables()

  return (
    //<SQLiteProvider databaseName='chatBot.db' useSuspense>
    <Tabs screenOptions={{ tabBarActiveTintColor: '#FFA001' ,tabBarShowLabel:false , tabBarInactiveTintColor:'#CDCDE0' , tabBarStyle:{
      backgroundColor: '#161622',
      borderTopWidth: 1,
      borderTopColor: '#232533',
      height:84,
      
    } 
    }}>
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
          headerShown:false,
        }}
      />

      
      
    </Tabs>
    //</SQLiteProvider>
  );
}

export default tabsLayout