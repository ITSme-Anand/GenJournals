import { View, Text } from 'react-native'
import React from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { GestureHandlerRootView,ScrollView } from 'react-native-gesture-handler'
import {images} from "../../constants";
import { Image } from 'react-native'
import FormField from '../../components/FormField';
import { useState } from 'react';
import CustomButton from "../../components/customButton"
import { Link, router } from 'expo-router';
import { signInWithEmailAndPassword } from 'firebase/auth';
import auth from '../../services/firebaseAuth';
import { getAuth } from 'firebase/auth';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useEffect } from 'react';
import { onAuthStateChanged } from 'firebase/auth';
const signIn = () => {



  const author = getAuth();
  const currentUser = author.currentUser;
  if(currentUser){
        router.push('/home');
  }
  const [form, setForm] = useState({
    email:'',
    password:'',
  })
    const checkLogIn = ()=>{
      onAuthStateChanged(auth,(user)=>{
          if(user){
              router.push('/home')
          }
      })
  }

  useEffect(()=>{
      checkLogIn()
  })


  const [isSubmitting, setIsSubmitting] = useState(false)
  const submit = ()=>{
    signInWithEmailAndPassword(auth,form.email,form.password)
    .then((userCred)=>{
      const user = userCred.user;
      router.push('/home')
      
    }).catch((error)=>{
        console.log('login attempt failed')
    })
  }
  return (
    
    <SafeAreaView className="bg-primary h-full">

         <GestureHandlerRootView>
            <ScrollView>
              <View className="w-full flex flex-column justify-start items-center px-4 my-6 min-h-[85vh]">
                <Image source={images.logo}
                      resizeMode='contain'
                      className='w-[400px] h-[200px]'

                />
                <Text className="text-2xl text-white text-semibold mt-10 font-psemibold"> Log In to GenJournals</Text>
                
                <FormField 
                    title='Email'
                    value={form.email}
                    handleChangeText={(e)=>setForm({...form,
                      email:e
                    })}
                    otherStyles="mt-7"
                    keyboardType="email-address"
                    placeholder="Enter your email address"
                />
                <FormField 
                    title='Password'
                    value={form.password}
                    handleChangeText={(e)=>setForm({...form,
                      password:e
                    })}
                    otherStyles="mt-7"
                    placeholder="Enter your password"
                />

                <CustomButton 
                    title="Sign In"
                    handlePress = {submit}
                    containerStyles='mt-7 px-10'
                    isLoading={isSubmitting}
                />
                <View className="justify-center pt-5 flex-row gap-2">
                    <Text className="text-lg text-gray-100 font-pregular">
                      Don't have an Account?{' '}
                      <Link href="/sign-up " className='text-lg font-psemibold text-secondary'>Sign Up</Link>
                    </Text>
                </View>
                
              </View>
            </ScrollView>
         </GestureHandlerRootView>
    </SafeAreaView>
  )
}

export default signIn