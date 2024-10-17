import { View, Text } from 'react-native'
import React from 'react'
import { Button } from 'react-native-paper';
import { Dialog, Portal, PaperProvider } from 'react-native-paper';


const logOutcomponent = () => {
    const [visible, setVisible] = React.useState(false);

    const showDialog = () => setVisible(true);
  
    const hideDialog = () => setVisible(false);
  return (
    <PaperProvider>
          <View>
            <Button
          className="bg-red-500 text-white text-lg w-[90vw] self-center mt-10"
          style={{borderRadius:6}}
          textColor='white'
          onPress={showDialog}
          
          >
              Logout
          </Button>
        <Portal>
          <Dialog visible={visible} onDismiss={hideDialog} className="bg-white">
            <Dialog.Title>Alert</Dialog.Title>
            <Dialog.Content>
              <Text variant="bodyLarge" className="text-white">This is simple dialog</Text>
            </Dialog.Content>
            <Dialog.Actions>
              <Button onPress={hideDialog}>Done</Button>
            </Dialog.Actions>
          </Dialog>
        </Portal>
        </View>
        </PaperProvider>
  )
}

export default logOutcomponent


