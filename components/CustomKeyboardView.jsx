import { View, Text , KeyboardAvoidingView , ScrollView , Platform} from 'react-native'
import React from 'react'
import { Keyboard } from 'react-native'
import { TouchableWithoutFeedback } from 'react-native'
const ios = Platform.OS =="ios"
const customKeyboardView = ({children , inChat}) => {
    let kavConfig = {};
    let scrollViewConfig = {};
    if(inChat){
        kavConfig = {keyboardVerticalOffset: 100}
        scrollViewConfig = {contentContainerStyle: {flex:1}}
    }
  return (
    <KeyboardAvoidingView
        behavior={ios?'padding':'height'}
        style={{flex:1}}
        keyboardVerticalOffset={-10}
        

    >
        
        <ScrollView 
            style={{flex:1}}
            bounces={false}
            showsVerticalScrollIndicator={false}
            {...scrollViewConfig}
        >
            <View style={{justifyContent: 'space-around',}}>
            {
                children
            }
            </View>
            
        </ScrollView>
        
        
    </KeyboardAvoidingView>
  )
}

export default customKeyboardView