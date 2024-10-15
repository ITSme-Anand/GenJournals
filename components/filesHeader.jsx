import { View, Text } from 'react-native'
import React from 'react'
import { IconButton, Searchbar } from 'react-native-paper'
import { useState } from 'react'
import { Button } from 'react-native-paper'
import { icons } from '../constants'
import { heightPercentageToDP as hp , widthPercentageToDP as wp } from 'react-native-responsive-screen'



const filesHeader = ({handleSearch, handleCreate, showButtons,handleDelete}) => {
    const [searchQuery, setSearchQuery] = React.useState('');
    
    return (
        <View className="bg-primary flex-col items-center" style={{paddingBottom:hp(2)}}>
                <View className="flex-row justify-end w-[95vw] bg-primary mb-2">
                        <IconButton
                            icon={icons.bin}
                            iconColor='#FFFFFF'
                            size={30}
                            onPress={handleDelete}
                            mode='contained'
                            className="bg-primary"
                            disabled={!showButtons}
                            
                            
                        
                        />
                        <IconButton
                            icon={icons.page}
                            iconColor='#FFFFFF'
                            size={30}
                            onPress={handleCreate}
                            mode='contained'
                            className="bg-primary"
                            
                            
                            
                        
                        />
                </View>
                <Searchbar
                    placeholder="Search"
                    placeholderTextColor={"#737373"}
                    onChangeText={setSearchQuery}
                    value={searchQuery}
                    onchangeText={handleSearch}
                    elevation={5}
                    inputStyle={{color:"#F0F0F0"}}
                    className="w-[95vw] bg-black-100"
                    iconColor='#737373'
                    style={{borderColor:"#737373"}}
                    
                />
                

        </View>
    )
    }

export default filesHeader