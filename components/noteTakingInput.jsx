import React from 'react';
import { TextInput, StyleSheet, View } from 'react-native';
import { heightPercentageToDP as hp , widthPercentageToDP as wp } from 'react-native-responsive-screen';
const NoteTakingInput = () => {
  return (
    <TextInput 
                
                value={text}
                onChangeText={text => setText(text)}
                inputMode='outlined'
                placeholder='Note'
                placeholderTextColor={'#737373'}
                multiline={true}
                textColor='white'
                className="bg-primary border-white"
                outlineStyle={{borderStyle:"dashed"}}
                selectionColor='#ADD8E6'
                cursorColor='#737373'
                underlineColor='#161622'
                activeUnderlineColor='#161622'
                
    />
  );
};



export default NoteTakingInput;
