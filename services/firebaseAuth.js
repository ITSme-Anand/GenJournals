// Import the functions you need from the SDKs you need
import { initializeApp , getApps } from "firebase/app";
import {initializeAuth , getReactNativePersistence ,getAuth} from 'firebase/auth';
import AsyncStorage from "@react-native-async-storage/async-storage";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyB4wl6-Vlyi88rp8avBZpw_Jc_dpYLM-lY",
  authDomain: "genjournals-5cae8.firebaseapp.com",
  projectId: "genjournals-5cae8",
  storageBucket: "genjournals-5cae8.appspot.com",
  messagingSenderId: "823673559463",
  appId: "1:823673559463:web:b22e772f9209afeeacfe82"
};
let auth;
if(getApps().length==0){
    // Initialize Firebase
    const app = initializeApp(firebaseConfig);
    auth = initializeAuth(app,{
    persistence: getReactNativePersistence(AsyncStorage)
});
}
else{
    auth = getAuth();
}


export default auth;