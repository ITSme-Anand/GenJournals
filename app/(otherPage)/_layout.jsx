
import {Stack} from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SQLiteProvider } from 'expo-sqlite';

const tabsLayout = () => {
  return (
    <>
      <SQLiteProvider databaseName='chatBot.db' useSuspense>
        <Stack>
            <Stack.Screen
              name="chatPage"
              options={{
                headerShown: false
              }} 
            
            />
            <Stack.Screen
              name="readFile"
              options={{
                headerShown: false
              }} 
            
            />
            
        </Stack>
        <StatusBar backgroundColor='#161622' style='light' hidden/>
      </SQLiteProvider>
      
    </>

    

  );
}

export default tabsLayout