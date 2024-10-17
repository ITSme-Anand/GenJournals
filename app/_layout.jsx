import { StyleSheet, Text, View } from 'react-native'
import React from 'react'
import { Slot , SplashScreen, Stack} from 'expo-router'
import { useFonts } from 'expo-font'
import { useEffect } from 'react'
import { SQLiteProvider } from 'expo-sqlite/next';
import loadDatabase from '../services/databaseIn'
import { useState } from 'react'

SplashScreen.preventAutoHideAsync();
const rootLayout = () => {
  const [fontsLoaded , error] = useFonts({
    "Poppins-Black": require("../assets/fonts/Poppins-Black.ttf"),
    "Poppins-Bold": require("../assets/fonts/Poppins-Bold.ttf"),
    "Poppins-ExtraBold": require("../assets/fonts/Poppins-ExtraBold.ttf"),
    "Poppins-ExtraLight": require("../assets/fonts/Poppins-ExtraLight.ttf"),
    "Poppins-Light": require("../assets/fonts/Poppins-Light.ttf"),
    "Poppins-Medium": require("../assets/fonts/Poppins-Medium.ttf"),
    "Poppins-Regular": require("../assets/fonts/Poppins-Regular.ttf"),
    "Poppins-SemiBold": require("../assets/fonts/Poppins-SemiBold.ttf"),
    "Poppins-Thin": require("../assets/fonts/Poppins-Thin.ttf"),
  })

  const [dbLoaded, setdbLoaded] = useState(false)

  useEffect(()=>{
    console.log('HELLO WE ARE IN THE ROOT LAYOUT')
    if(error) throw error;
    
    if(fontsLoaded) SplashScreen.hideAsync();
    loadDatabase()
    .then(
      ()=>{
        setdbLoaded(true)
        console.log('database loaded')
      }
    )
  .catch((e) => console.error(e));

  } , [fontsLoaded,error])

  
  

  if(!fontsLoaded && !error) return null;
  return(
    dbLoaded?(
    <SQLiteProvider  databaseName='chatBot.db' useSuspense>
    <Stack>
        <Stack.Screen name="index" options={{headerShown:false}} /> 
        <Stack.Screen name="(auth)" options={{headerShown:false}} /> 
        <Stack.Screen name="(tabs)" options={{headerShown:false}} /> 
        <Stack.Screen name="(otherPage)" options={{headerShown:false , navigationBarHidden:true}} />
    </Stack>
    </SQLiteProvider>):(<></>)
    
  )
}

export default rootLayout
