import { ActivityIndicator, Platform, Text, View } from "react-native";
import * as FileSystem from "expo-file-system";
import { Asset } from "expo-asset";
import React from "react";

const loadDatabase = async () => {
    const dbName = "chatBot.db";
    const dbAsset = require("./../assets/chatBot.db");
    const dbUri = Asset.fromModule(dbAsset).uri;
    const dbFilePath = `${FileSystem.documentDirectory}SQLite/${dbName}`;
    const fileInfo = await FileSystem.getInfoAsync(dbFilePath);
    if (!fileInfo.exists) {
      await FileSystem.makeDirectoryAsync(
        `${FileSystem.documentDirectory}SQLite`,
        { intermediates: true }
      );
      await FileSystem.downloadAsync(dbUri, dbFilePath);
    }
  };



export default loadDatabase;