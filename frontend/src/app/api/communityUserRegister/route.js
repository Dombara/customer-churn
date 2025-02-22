import {dbconnect} from "../../../utils/dbConnect/dbConnect";
import CommunityUser from '@/utils/schema/communityUserSchema';
// import {NextResponse} from "next/server"
dbconnect();



export async function POST(req) {
  const today = new Date()
    try {
      // Parse JSON body and destructure the form data
      const { fullName, providerType, orgName, email, phone } = await req.json();
  
      // Example: Log or handle the form data (e.g., save to a database)
      console.log('Received data:', { fullName, providerType, orgName, email, phone });

      // Save the data to the database
      const newCommunityUser = new CommunityUser({
        fullName,
        providerType,
        orgName,
        email,
        phone,
        createdAt: today,
      });
      await newCommunityUser.save();
      return new Response(JSON.stringify({ success: true, message: "Registration successful" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
  
      // // Return a success response
      // return new Response(JSON.stringify({ success: true, message: "Registration successful" }), {
      //   status: 200,
      //   headers: { "Content-Type": "application/json" },
      // });
    } catch (error) {
        console.error("Error in registration:", error);
        return new Response(JSON.stringify({ error: "Server error. Please try again." }), {
            status: error.statusCode || 500,
            headers: { "Content-Type": "application/json" },
        });
    }
  }
  



