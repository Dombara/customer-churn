'use client';
import { useAuth } from '../../../context/AuthContext';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function ProviderLogin() {

  const { loginCommunity } = useAuth();
  const router = useRouter();

  const [formData, setFormData] = useState({
    fullName: '',
    providerType: '',
    orgName: '',
    email: '',
    phone: '',
    // documentType: '',
  });

  // const handleLogin = (e) => {
  //   e.preventDefault();
  //   // Mock authentication
  //   loginCommunity({ name: 'Community User' });
  //   router.push('/community/communityDashboard');
  // };

  const communityUserRegister = async(e) => { 
    e.preventDefault();
    // Mock authentication
    // send api request to backend
    // Send data to the API
    try {
      const response = await fetch('/api/communityUserRegister', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();

      if (data.success) {
          setFormData({ fullName: '', providerType: '', orgName: '', email: '', phone: '' });
          alert(data.message);
          loginCommunity({ name: formData.fullName, providerType: formData.providerType, orgName: formData.orgName, email: formData.email, phone: formData.phone });
          router.push('/community/communityDashboard');
      } else {
          alert('Registration failed');
          console.log('Error:', data.message);
      }
    }
    catch (error) {
      alert('Could not Save data. Please try again later! :');
      console.log('Error : ', error);
    }
  }

  

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 className="text-3xl font-semibold text-center mb-6">Community Login</h1>

        <form className="space-y-6">
          {/* Full Name Field */}
          <div className="flex flex-col gap-2">
            <label htmlFor="fullName" className="font-medium">Full Name</label>
            <input
              type="text"
              id="fullName"
              value={formData.fullName}
              onChange={(e)=>setFormData({...formData, fullName: e.target.value})}  
              placeholder="Enter your full name"
              className="border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Select Provider Type */}
          <div className="flex flex-col gap-2">
            <label className="font-medium">Select Provider Type</label>
            <div className="flex justify-start gap-4">
              <button type="button" className="p-2 flex items-center gap-2 bg-gray-100 rounded-md hover:bg-gray-200">
                <span>😊</span>
                <span>Volunteer</span>
              </button>
              <button type="button" className="p-2 flex items-center gap-2 bg-gray-100 rounded-md hover:bg-gray-200">
                <span>🏛️</span>
                <span>Government</span>
              </button>
              <button type="button" className="p-2 flex items-center gap-2 bg-gray-100 rounded-md hover:bg-gray-200">
                <span>🌐</span>
                <span>Other</span>
              </button>
            </div>
          </div>

          {/* Organization Name Field */}
          <div className="flex flex-col gap-2">
            <label htmlFor="orgName" className="font-medium">Organization Name</label>
            <input
              type="text"
              id="orgName"
              placeholder="Enter your organization name"
              className="border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={formData.orgName}
              onChange={(e)=>setFormData({...formData, orgName: e.target.value})}
            />
          </div>

          {/* Email Field */}
          <div className="flex flex-col gap-2">
            <label htmlFor="email" className="font-medium">Email</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email"
              className="border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={formData.email}
              onChange={(e)=>setFormData({...formData, email: e.target.value})}
            />
            <span className="text-sm text-gray-500">Used for verification</span>
          </div>

          {/* Phone Field */}
          <div className="flex flex-col gap-2">
            <label htmlFor="phone" className="font-medium">Phone</label>
            <input
              type="tel"
              id="phone"
              placeholder="Enter your phone number"
              className="border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={formData.phone}
              onChange={(e)=>setFormData({...formData, phone: e.target.value})}
            />
          </div>

          {/* Location Field */}
          {/* <div className="flex flex-col gap-2">
            <label htmlFor="location" className="font-medium">Location</label>
            <input
              type="text"
              id="location"
              placeholder="Enter your location"
              className="border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div> */}

          {/* Document Type Field */}
          <div className="flex flex-col gap-2">
            <label className="font-medium">Document Type</label>
            <div className="flex gap-4">
              <button type="button" className="p-2 flex items-center gap-2 bg-gray-100 rounded-md hover:bg-gray-200">
                ID Badge
              </button>
              <button type="button" className="p-2 flex items-center gap-2 bg-gray-100 rounded-md hover:bg-gray-200">
                Organization Letter
              </button>
            </div>
            <span className="text-sm text-gray-500">Upload proof of affiliation</span>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-6 justify-center mt-6">
            {/* <button type="button" className="border border-gray-300 rounded-md py-3 px-6 text-lg font-semibold text-gray-700 hover:bg-gray-100">
              Login
            </button> */}
            <button type="button" className="bg-blue-500 text-white rounded-md py-3 px-6 text-lg font-semibold hover:bg-blue-600" onClick={communityUserRegister}>
              Register
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
