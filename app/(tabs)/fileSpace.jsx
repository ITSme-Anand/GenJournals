import { View, Text } from 'react-native'
import React from 'react'
import { FlatList } from 'react-native'
import { useRouter } from 'expo-router'
import { useState } from 'react'
import { SafeAreaView } from 'react-native'
import FileItem from '../../components/fileItem'
import { useEffect } from 'react'
import * as SQLite from 'expo-sqlite';
import FilesHeader from '../../components/filesHeader'
import { heightPercentageToDP as hp , widthPercentageToDP as wp } from 'react-native-responsive-screen'
import loadDatabase from '../../services/databaseIn'
import { TouchableWithoutFeedback } from 'react-native'
import { useFocusEffect } from 'expo-router'
import { useCallback } from 'react'
import { KeyboardAvoidingView } from 'react-native'
import { Platform } from 'react-native'

//if I uncomment the create and delete file functionalities it is working now.

const fileSpace = () => {
  //const documents = [{'id':1,'title':'gratitudeJournal','tableName':`userFile1`},{'id':2,'title':'Goals','tableName':`userFile2`},{'id':3,'title':'Journaling','tableName':'userFile3'}]
  const [files , setFiles] = useState(null);
  const [selectedFile, setSelectedFile] = useState({'id':0})
  const Router = useRouter();
  const [newFileID, setNewFileID] = useState(1)
  const [showbuttons, setShowbuttons] = useState(false)
  const [longPressOff, setlongPressOff] = useState(true)
  console.log('files is initially' , files)
  console.log('newFileId is', newFileID);

  //fetching all the tables
  useFocusEffect(
    useCallback(() => {
      // Re-fetch or update when screen comes into focus
      fetchAllTables();
    }, [])
  );
  
  //db opening
  const db = SQLite.useSQLiteContext()
  

  useEffect(()=>{
    fetchAllTables();
  } , [])


  //this is called whenever a file is created or deleted and each time we come out of the readFile component back to the fileSpace
  const fetchAllTables = async ()=>{
    try{
    
    
      const allFileDetails = await fetchFileDetails();
      console.log('all the files are:', allFileDetails)
      const len = allFileDetails.length
      console.log('length of all files is', len)
      console.log('last file is:',allFileDetails[len-1])
      const a = allFileDetails[len-1]['primaryID']
      console.log('a value is',a)
      //NewFileID should be a+1
      setFiles(allFileDetails)
      setNewFileID(a+1)//here is the issue. you can't set it like that. you have to set it to the 
    }
    catch(err){
      console.log('error happened here')
      console.error(err)
    }
    
    
  }

  //called only by the fetchFile detail function
  const fetchFileDetails = async ()=>{
    let result;
    try{
      console.log('inside try')
      result = await db.getAllAsync(`SELECT * FROM tableMetaData;`)
    }
    catch(err){
      console.log('error has occured')
      console.error(err)
    }
    console.log(result)
    console.log('about to return')
    return result;
  }

  //does not run in transaction. this function should be called only once. maybe try to keep this function independent of all the components
  /* const createTablesAtFirst = async ()=>{
    try{//create a primary id for all the userFile tables
       DROP TABLE IF EXISTS userFile1;
      DROP TABLE IF EXISTS userFile2;
      DROP TABLE IF EXISTS tableMetaData;
      INSERT INTO userFile1 (id,entry) VALUES (1,'Gratitude Journal');
      INSERT INTO userFile1 (id,entry) VALUES (2,'');
      INSERT INTO userFile2 (id,entry) VALUES (1,'Goal Trackers');
      INSERT INTO userFile2 (id,entry) VALUES (2,'');
      INSERT INTO tableMetaData (id,tableName,title,line1,line2, line3) VALUES (1,'userFile1','Gratitude Journal','','','');
      INSERT INTO tableMetaData (id,tableName,title,line1,line2, line3) VALUES (2,'userFile2','Goal Trackers','','','');
      await db.execAsync(
        `
        CREATE TABLE IF NOT EXISTS userFile1 (
        primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        entry TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS userFile2 (
        primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        entry TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS tableMetaData (
        primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        tableName TEXT NOT NULL,
        title TEXT,
        line1 TEXT,
        line2 TEXT,
        line3 TEXT
        );
        
        `
      )
    }catch(err){
      console.error('error while creating permanent tables')
      console.error(err)
    }
    
  } */

  //functions for handling creation and searching
  const createFile = async ()=>{
    console.log('inside createFile function')
    //the newFileID must be a new row's Primary-ID. if I solve that , then the bug is solved
    let id = newFileID;
    let tableName = `userFile${newFileID}`;
    console.log('the new file id is' , newFileID)
    console.log('the new id is ',id)
    console.log('the new table name is', tableName)
    let newDocObj = {'id':id, 'tableName':tableName , 'title':'','line1':'','line2':'','line3':''}
    try{
      await db.execAsync(//here also you need to add a primary-id column and you have insert id manually
        `
          CREATE TABLE IF NOT EXISTS ${tableName} (
          primaryID INTEGER PRIMARY KEY AUTOINCREMENT,
          id INTEGER NOT NULL,
          entry TEXT,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
          );
          INSERT INTO ${tableName} (primaryID,id,entry) VALUES (1,1,'');
          INSERT INTO ${tableName} (primaryID,id,entry) VALUES (2,2,'');
          INSERT INTO tableMetaData (id,tableName,title,line1,line2,line3) VALUES (${newFileID},'${tableName}','','','','');
          
        `
      ) 
      
     
      
    }
    catch(err){
        console.log('error here')
        console.error(err)
    }
    console.log('file created')
    await fetchAllTables();
    
    

  }
  const handleDelete = async ()=>{
    console.log('inside handle delete')
    if(!selectedFile){
      return
    }
    console.log('file to be deleted is:')
    console.log(selectedFile)
    const tableName = selectedFile['tableName'];
    try{await db.execAsync(
      `
        DROP TABLE IF EXISTS ${tableName};
        DELETE FROM tableMetaData WHERE tableName = '${tableName}';
      `
    )
    }
    catch(err){
      console.error(err)
    }
    await fetchAllTables();
    setShowbuttons(false)
    setSelectedFile({'id':0})
  }


  //functions that dont interact with sqlite
  const handleSearch = ()=>{
    console.log('search Is Handled')
  }
  const handleLongPress = (fileDetails)=>{
    console.log('button is long pressed')
    console.log('file details are:')
    console.log(fileDetails)
    setSelectedFile(fileDetails)
    setlongPressOff(false)
    setShowbuttons(true)

  }
  const handleOutsidePress = ()=>{
    setShowbuttons(false)
    setlongPressOff(true)
    setSelectedFile({'id':0})
    
  }
  return (
    <TouchableWithoutFeedback onPress={handleOutsidePress}>
        <View className="flex-1 bg-primary">
          <View className="flex-1" style={{paddingTop: hp(6)}}>
            <FilesHeader handleSearch={handleSearch} handleCreate={createFile} showButtons={showbuttons} handleDelete={handleDelete}/>
            <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={{ flex: 1 }}>
            <View style={{flex:1}}>
              <FlatList 

                  data={files}
                  keyExtractor={item=>item['id']}
                  showsVerticalScrollIndicator={false}
                  renderItem={({item, index})=><FileItem updateFileRendering={fetchAllTables} details={item} index={index} router={Router} handleLongPress={handleLongPress} longPressOff={longPressOff} selectedFile={selectedFile}  />}
                  contentContainerStyle={{flexDirection:'column', alignItems:'center' }}
                  className="bg-primary"
                  scrollEnabled={true}

              />
            </View>
            </KeyboardAvoidingView>
        </View>
      </View>
    </TouchableWithoutFeedback>
    
    
  )
}

export default fileSpace