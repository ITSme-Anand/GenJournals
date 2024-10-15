import { View } from 'react-native'
import React from 'react'
import { TouchableOpacity } from 'react-native'
import { Avatar, Button, Card, Text } from 'react-native-paper';


const fileItem = ({updateFileRendering , details , index , router, handleLongPress, selectedFile}) => {
    const goToFile = ()=>{
        router.push({pathname:'/readFile',params:details})
    }
    const line1 = !(details['line1']=='');
    const line2 = !(details['line2']=='');
    const line3 = !(details['line3']=='');
    /* console.log('selected file is')
    console.log(selectedFile)
    console.log('actual file is')*/
    //console.log(details) 
    
    
  return (
            
                <View>
                  <Card
                        
                        onPress={goToFile} 
                        onLongPress={()=>
                          {
                              
                              handleLongPress(details)
                              
                              
                          }}
                        
                        className="w-[95vw] bg-primary py-4 my-2 border border-gray-700 rounded-xl"
                        style={selectedFile['id']==details['id']?{borderColor:"#FFFFFF"}:null}
                    >
                      <Text className="text-white text-lg mb-1"> {details['title']} </Text>
                      {(line1|line2|line3)?(
                            <Card.Content>
                            <Text className="text-white" variant="bodyMedium">{details?details['line1']:'loading'}</Text>
                            <Text className="text-white" variant="bodyMedium">{details?details['line2']:'loading'}</Text>
                            <Text className="text-white" variant="bodyMedium">{details?details['line3']:'loading'}</Text>
                          </Card.Content>
                  ):(
                    <></>
                  )}
                  </Card>
                
                </View>
                

            
    
  )
}

export default fileItem