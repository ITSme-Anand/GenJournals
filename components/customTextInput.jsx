import { View, Text } from 'react-native'
import React from 'react'
import { TextInput } from 'react-native-paper'
import { useState } from 'react'
import { useEffect } from 'react'
import { useRef } from 'react'
const customTextInput = ({ fileDetails , entry , placeholder, handleKeyPress , index , text , handleSave , focusedId}) => {
    
    const [TEXT, setTEXT] = useState(text)
    const inputRef = useRef(null)
    
    useEffect(() => {
      // Automatically focus the input when the component mounts
      
      if(focusedId){
        console.log('entry is' , entry)
        inputRef.current?.focus(); // Check if ref is defined and call focus
      }
      
    }, []);
  return (
    <TextInput 
                
                value={TEXT}
                ref={inputRef}
                onChangeText={(text) =>{
                    
                    setTEXT(text)
                    handleSave(fileDetails , entry , text)
                }}
                inputMode='outlined'
                placeholder={placeholder}
                placeholderTextColor={'#737373'}
                multiline={true}
                dense
                textColor='white'
                className={`bg-primary py-0 ${placeholder==='Title'?'text-xl font-bold':''}`}
                focusable={placeholder==='Title'}
                selectionColor='#ADD8E6'
                cursorColor='#737373'
                underlineColor='#161622'
                activeUnderlineColor='#161622'
                
                onKeyPress={({nativeEvent})=>{
                  handleKeyPress(nativeEvent, entry)
                  setTEXT(TEXT)
                }}
                
                
    />
  )
}

export default customTextInput