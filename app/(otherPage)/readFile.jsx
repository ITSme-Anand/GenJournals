import { View, Text } from 'react-native'
import React from 'react'
import { useLocalSearchParams, useRouter } from 'expo-router';
import FileRoomHeader from '../../components/fileRoomHeader';
import { SafeAreaView } from 'react-native';
import { heightPercentageToDP as hp , widthPercentageToDP as wp } from 'react-native-responsive-screen';
import CustomTextInput from '../../components/customTextInput';
import { useState } from 'react';
import * as SQLite from 'expo-sqlite';
import { useEffect } from 'react';
import loadDatabase from '../../services/databaseIn';
import { FlatList } from 'react-native';
const readFile = () => {
  const router = useRouter();
  const [text, setText] = useState("");
  const [allData, setAllData] = useState(null)
  const [dataLoaded, setdataLoaded] = useState(false)
  const [newEntryID, setnewEntryID] = useState(0)
  const [currentlyFocused, setcurrentlyFocused] = useState(1)
  
  const db = SQLite.useSQLiteContext()
 
  let details;
  try{
    details = useLocalSearchParams(details)
    //console.log(details)
  }
  catch(err){
    console.error(err)
  }

  useEffect(()=>{
    db.withTransactionAsync(async ()=>{
      try{
        const results = await loadData();
        setAllData(results)
        setdataLoaded(true)
        console.log('data loaded')
        console.log(results)
        let max=0;
        let len=0;
        results.forEach(
          (item)=>{
            if(item['id']>max){
              max = item['id']
              len=len+1;

            }
            
          }

        )
        
        setnewEntryID(max+1)
        
      }
      catch(err){
        console.error(err)
      }
      
    })
  } , [db])

  const loadData = async ()=>{
    console.log('inside the loadData')
    console.log(details)
    console.log('the table name is',details["tableName"])
    const tableName = details['tableName'];
    try{
      let results = await db.getAllAsync(
      `SELECT * FROM '${tableName}';`
    )
    console.log('all the data are loaded from the table')
    let len = results.length
    console.log('currently focused is',results[len-1]['id'])
    setcurrentlyFocused(results[len-1]['id'])
    //console.log(results)
    
    
    return results}
    catch(err){
      console.log('error while loading data')
      console.error(err)
    }
  }


  const handleBackButton = async (fileDetails) =>{
    const tableName = fileDetails['tableName']
    
    try {
      await db.withTransactionAsync(async () => {
        // Prepare your insert statement
        await db.execAsync(
          `DELETE FROM ${tableName};`
        )
        const statement = await db.prepareAsync(
          `INSERT INTO ${tableName} (id, entry,timestamp) VALUES ($id, $entry, CURRENT_TIMESTAMP)`
        );
  
        for (let detail of allData) {
          const { entry ,id ,primaryID, timestamp} = detail;
          let result = await statement.executeAsync({ $id: id, $entry: entry });
          console.log( result.lastInsertRowId, result.changes);
        }
        const title = allData[0]?allData[0]['entry']:'';
        const line1 = allData[1]?allData[1]['entry']:'';
        const line2 = allData[2]?allData[2]['entry']:'';
        const line3 = allData[3]?allData[3]['entry']:'';
        
        await db.execAsync(
          `UPDATE tableMetaData SET title='${title}',line1='${line1}',line2='${line2}',line3='${line3}' WHERE tableName = '${tableName}';
          
          `
          
        )
        console.log('committed')
      });
  
      console.log('Data inserted successfully');
    } catch (error) {
      console.error('Error inserting data:', error);
    }
  };




  //the functions which are not interacting with sql

  
  const handleKeyPress = (nativeEvent, entry) => {
    if(nativeEvent['key']=="Enter"){
      console.log('enter is pressed')
      console.log('entry is ', entry)
      console.log('the new entry to be created is of id ', newEntryID);
      const index = allData.findIndex((item) => item['id'] === entry['id']);
      let newEntries = allData.map(item=>item);
      newEntries.splice(index+1 , 0 , {"entry":"","id":newEntryID,"timestamp":""})
      console.log(newEntries)
      setcurrentlyFocused(newEntryID)
      setnewEntryID(newEntryID+1)
      setAllData(newEntries)
    }
  };

  const saveTheData = (fileDetails , entryDetails , newEntry)=>{
    //what this function is doing? whenever an input box text is changed , we will change it here
    entryDetails['entry'] = newEntry.trimEnd();
    console.log('new entry is', entryDetails)
    const FileEntries = allData.map((item)=>{
      if(item['id']==entryDetails['id']){
        return entryDetails
      }
      else{
        return item
      }
    })
    console.log('new file entries are:' , FileEntries);
    setAllData(FileEntries)
  }


  return (
    <SafeAreaView className="bg-primary h-full">
        <FileRoomHeader fileDetails={details} handlePressBack={handleBackButton} router={router}/>
          <View className="p-2 mb-2 pt-2 pl-2 mx-1">
            {dataLoaded?(
              <FlatList 
                data={allData}
                keyExtractor={item=>item['id']}
                showsVerticalScrollIndicator={false}
                renderItem={({item, index})=><CustomTextInput focusedId={currentlyFocused==item['id']?true:false} fileDetails={details} entry={item} handleSave={saveTheData} index={index} handleKeyPress={handleKeyPress} text={item['entry']} placeholder={item['id']==1?'Title':'Note'}/>}

              />
            ):(<View></View>)}
              
              
          </View>
        
    </SafeAreaView>
  )
}

export default readFile