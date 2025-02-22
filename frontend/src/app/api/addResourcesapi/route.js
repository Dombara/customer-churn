// pages/api/submitResource.js
import {dbconnect} from "../../../utils/dbConnect/dbConnect";
import Resource from '@/utils/schema/resourceSchema';

export async function POST(req) {
    const today = new Date();
    try {
      const { location, category, capacity, contactInfo, operatingHours, specialServices, notes } = await req.json();
      console.log('Received data:', { location, category, capacity, contactInfo, operatingHours, specialServices, notes });
  
      await dbconnect(); // Connect to the database
      
      const newResource = new Resource({
        location,
        category,
        capacity,
        contactInfo,
        operatingHours,
        // specialServices,
        notes,
        createdAt: today,
      });
  
      const savedResource = await newResource.save();
      return new Response(
        JSON.stringify({ success: true, message: 'Resource added successfully'}),
        { status: 200, headers: { 'Content-Type': 'application/json' }}
      );
    } catch (error) {
      console.error("Error adding resource:", error);  // This will log the error in the server
      return new Response(
        JSON.stringify({ error: 'Server error. Please try again.'}),  // Include more error details
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }
  }
  
