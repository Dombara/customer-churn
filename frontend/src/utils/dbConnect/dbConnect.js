import mongoose from 'mongoose';

const connection ={}

export async function dbconnect(){
    if(connection.isConnected){
        console.log('Using Existing Database Connection');
        return;
    }
    try{
        const db = await mongoose.connect(process.env.MONGODB_URI || '',{})
        connection.isConnected = db.connections[0].readyState;
        console.log('DB Coonection Successful');

    }catch(error){
        console.log("Database connection Failed : ", error);
        process.exit(1);
    }
}