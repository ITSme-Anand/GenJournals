import { StatusBar } from "expo-status-bar";
import { Text,View } from "react-native";
import { Link } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { GestureHandlerRootView, ScrollView } from "react-native-gesture-handler";
import { Image } from "react-native";
import {images} from "../constants";
import CustomButton from "../components/customButton";
import { router } from "expo-router";
import { onAuthStateChanged } from "firebase/auth";
import auth from "../services/firebaseAuth";
import { useEffect } from "react";
export default function App(){

    const checkLogIn = ()=>{
        onAuthStateChanged(auth,(user)=>{
            if(user){
                router.push('/home')
            }
        })
    }

    useEffect(
        ()=>{
            checkLogIn();
        }
    )
    
    return(
        <SafeAreaView className="bg-primary h-full">
            <GestureHandlerRootView>
                <ScrollView contentContainerStyle={{height:'100%'}}>
                    <View className="w-full min-h-[85vh] flex-col justify-start items-center px-4">
                        <Image source={images.logo} className="h-[200px] w-[400px]" resizeMode="contain"/>
                        <Image source={images.cards} className="max-w--[380px] w-full h-[250px]" resizeMode="contain"/>
                        <View className="relative mt-5">
                            <Text className="text-1xl text-white font-bold text-center py-0 my-0"> 
                                Enhance Your Mental health with
                                <Text className="text-secondary-200 text-2xl">&nbsp; GenJournals</Text>
                            </Text>
                            <Image source={images.path} className=" w-[236px] h-[28px] absolute -bottom-4 right-2" resizeMode="center"/>
                        </View>
                        <Text className="text-sm font-pregular text-gray-100 mt-2"> Unlock Your True Inner Potential</Text>
                        <CustomButton 
                        title="Continue with Email"
                        handlePress = {()=>router.push('/sign-up')}
                        containerStyles="w-full mt-7"
                        textStyles="px-4"/>
                    </View>
                    
                </ScrollView>
            </GestureHandlerRootView>
            <StatusBar   backgroundColor="#161622" style="light" />
        </SafeAreaView>
    )
}

 