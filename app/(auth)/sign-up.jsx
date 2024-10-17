import { View, Text } from 'react-native'
import React from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { GestureHandlerRootView,ScrollView } from 'react-native-gesture-handler'
import {images} from "../../constants";
import { Image } from 'react-native'
import FormField from '../../components/FormField';
import { useState } from 'react';
import CustomButton from "../../components/customButton"
import { Link } from 'expo-router';
import {createUserWithEmailAndPassword} from 'firebase/auth'
import auth from '../../services/firebaseAuth';
import AsyncStorage, { useAsyncStorage } from '@react-native-async-storage/async-storage';
import { getAuth } from 'firebase/auth';
import { router } from 'expo-router';
import { useEffect } from 'react';
import { signOut, onAuthStateChanged } from 'firebase/auth';
const signUp = () => {
  const [form, setForm] = useState({
    userName:'',
    email:'',
    password:'',
  })

  const [emailError, setEmailError] = useState(null)
  const [passwordError, setPasswordError] = useState(null)



  const [isSubmitting, setIsSubmitting] = useState(false)
  const submit = ()=>{
    setEmailError(null);
    setPasswordError(null);
    createUserWithEmailAndPassword(auth,form.email,form.password)
    .then((userCredentials)=>{
      const user = userCredentials.user;
      AsyncStorage.setItem('userName', form.userName)
      .catch((err)=>{
        console.log(error)
      });
      AsyncStorage.setItem('AIfriendName', 'AI Friend').catch(err=>console.log(err))
      AsyncStorage.setItem('futureSelfName','Future-Self').catch(err=>console.log(err))
      const author = getAuth();
      const currentUser = author.currentUser;
      if(currentUser){
        router.push('/home');
      }


    })
    .catch((error)=>{
      console.log(error)
      if(error.code==="auth/email-already-in-use"){
        setEmailError("Email Already Registered!")
      }
      else if(error.code==="auth/invalid-email"){
        setEmailError("Email is invalid!")
      }
      else if(error.code==="auth/missing-email"){
        setEmailError("Email is missing!")
      }
      if(error.code==="auth/weak-password"){
        setPasswordError("Password not long than 6 characters!")
      }
      if(error.code==="auth/missing-password"){
        setPasswordError("Password is empty")
      }
      if(error.code==="auth/invalid-password"){
        setPasswordError("Password is invalid")
      }
    })
  }
  return (
    
    <SafeAreaView className="bg-primary h-full">

         <GestureHandlerRootView>
            <ScrollView>
              <View className="w-full flex flex-column justify-start items-center px-4 my-6 min-h-[85vh]">
                <Image source={images.logo}
                      resizeMode='contain'
                      className='w-[400px] h-[100px]'

                />
                <Text className="text-2xl text-white text-semibold mt-10 font-psemibold"> Sign Up to GenJournals</Text>
                
                <FormField 
                    title='UserName'
                    value={form.userName}
                    handleChangeText={(e)=>setForm({...form,
                      userName:e
                    })}
                    otherStyles="mt-10"
                    placeholder="Enter your UserName"
                />
                

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
                {emailError&&(<Text className=" text-red-600 text-semibold">{emailError}</Text>)}
                <FormField 
                    title='Password'
                    value={form.password}
                    handleChangeText={(e)=>setForm({...form,
                      password:e
                    })}
                    otherStyles="mt-7"
                    placeholder="Enter your password"
                />
                {passwordError&&(<Text className=" text-red-600 text-semibold">{passwordError}</Text>)}
                <CustomButton 
                    title="Sign Up"
                    handlePress = {submit}
                    containerStyles='mt-7 px-10'
                    isLoading={isSubmitting}
                />
                <View className="justify-center pt-5 flex-row gap-2">
                    <Text className="text-lg text-gray-100 font-pregular">
                      Already have an account?{' '}
                      <Link href="/sign-in " className='text-lg font-psemibold text-secondary'>Log In</Link>
                    </Text>
                </View>
                
              </View>
            </ScrollView>
         </GestureHandlerRootView>
    </SafeAreaView>
  )
}

export default signUp