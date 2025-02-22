"use client";
import React, { useState } from 'react';
// import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
// import L from 'leaflet';
// import 'leaflet/dist/leaflet.css';

const SubmitResources = () => {
  const [formData, setFormData] = useState({
    location: '',
    // category: [],
    category: '',
    capacity: '',
    contactInfo: '',
    operatingHours: '',
    // status: 'Available',
    // specialServices: { wheelchair: false },
    // specialServices: '',
    notes: '',
    // photos: [],
  });

//   const categories = [
//     { name: 'Emergency Shelters', icon: '🏠' },
//     { name: 'Medical Facilities', icon: '🏥' },
//     { name: 'Food and Water Supply', icon: '🍽️' },
//     { name: 'Volunteer Organizations', icon: '🤝' },
//     { name: 'Evacuation Routes', icon: '🚗' },
//     { name: 'Utility and Power Stations', icon: '🔋' },
// ];


  const [position, setPosition] = useState([51.505, -0.09]);
//   const [marker, setMarker] = useState(null);

  const handleGetLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setPosition([latitude, longitude]);
          setFormData({ ...formData, location: `${latitude}, ${longitude}` });
        },
        (error) => console.error("Error getting location:", error)
      );
    } else {
      console.error("Geolocation is not supported by this browser.");
    }
  };


//   const MapEvents = () => {
//     const map = useMapEvents({
//       click(e) {
//         const { lat, lng } = e.latlng;
//         setPosition([lat, lng]);
//         setFormData({ ...formData, location: `${lat}, ${lng}` });
//         if (marker) {
//           marker.setLatLng([lat, lng]);
//         } else {
//           const newMarker = L.marker([lat, lng]).addTo(map);
//           setMarker(newMarker);
//         }
//       },
//     });
//     return null;
//   };

  // Handle category changes
//   const handleCategoryChange = (category) => {
//     const newCategories = formData.category.includes(category)
//       ? formData.category.filter((item) => item !== category)
//       : [...formData.category, category];
//     setFormData({ ...formData, category: newCategories });
//   };

//   const customIcon = L.icon({
//     iconUrl: '/community/custom-marker-icon.png', // Path to your custom icon
//     shadowUrl: '/community/custom-marker-shadow.png', // Path to your custom shadow image
//   });
  
//   L.marker([latitude, longitude], { icon: customIcon }).addTo(map);
  
    // Handle form submission
    const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch('/api/addResources', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    });

    const data = await response.json();
    if (response.ok) {
        alert('Resource submitted successfully!');
    } else {
        alert(`Error: ${data.message}`);
    }
    };


  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50">
      <div className="w-full max-w-3xl p-8 bg-white rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-6">Add Resource</h1>
        
        <form className="space-y-8">
          {/* Location */}
          <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">📍</span> Location
            </label>
            <input
              type="text"
              placeholder="Enter address or select location on map"
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
              className="p-3 border border-gray-300 rounded-lg w-full focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none"
            />
            {/* <div className="h-48 border rounded-md overflow-hidden mx-auto mt-4" style={{ width: '100%', maxWidth: '400px' }}>
              <MapContainer center={position} zoom={13} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; OpenStreetMap contributors'
                />
                <Marker position={position} />
                <MapEvents />
              </MapContainer>
            </div> */}
            <button
              type="button"
              onClick={handleGetLocation}
              className="mt-4 bg-blue-500 text-white py-2 px-4 rounded-md shadow hover:bg-blue-600 transition duration-300"
            >
              📍 Use My Location
            </button>
          </div>

          {/* Categories */}
          <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">📋</span> Resource Category
            </label>
            {/* <div className="grid grid-cols-2 gap-4">
              {[
                { name: 'Emergency Shelters', icon: '🏠' },
                { name: 'Medical Facilities', icon: '🏥' },
                { name: 'Food and Water Supply', icon: '🍽️' },
                { name: 'Volunteer Organizations', icon: '🤝' },
                { name: 'Evacuation Routes', icon: '🚗' },
                { name: 'Utility and Power Stations', icon: '🔋' },
              ].map((category) => (
                <label key={category.name} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={formData.category.includes(category.name)}
                    onChange={() => handleCategoryChange(category.name)}
                    className="form-checkbox h-5 w-5 text-blue-500"
                  />
                  <span className="text-gray-700 font-medium flex items-center">
                    <span className="mr-2">{category.icon}</span>{category.name}
                  </span>
                </label>
              ))}
            </div> */}
            <div className="grid grid-cols gap-4">
            
            <input
              type="text"
              placeholder="Enter Resource category"
              value={formData.category}
              onChange={(e) => setFormData({ ...formData, category: e.target.value })}
              className="p-3 border border-gray-300 rounded-lg w-full focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none"
            />
            </div>
        
          </div>

          {/* Capacity */}
          <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">🔢</span> Capacity
            </label>
            <input
              type="text"
              placeholder="Enter current/max capacity"
              value={formData.capacity}
              onChange={(e) => setFormData({ ...formData, capacity: e.target.value })}
              className="p-3 border border-gray-300 rounded-lg w-full focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none"
            />
          </div>

          {/* Contact Info */}
          <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">📞</span> Contact Information
            </label>
            <input
              type="text"
              placeholder="Enter contact info (phone, email)"
              value={formData.contactInfo}
              onChange={(e) => setFormData({ ...formData, contactInfo: e.target.value })}
              className="p-3 border border-gray-300 rounded-lg w-full focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none"
            />
          </div>

          {/* Operating Hours */}
          <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">🕒</span> Operating Hours
            </label>
            <input
              type="text"
              placeholder="Enter operating hours"
              value={formData.operatingHours}
              onChange={(e) => setFormData({ ...formData, operatingHours: e.target.value })}
              className="p-3 border border-gray-300 rounded-lg w-full focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none"
            />
          </div>

          {/* Special Services */}
          {/* <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">♿</span> Special Services
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.specialServices.wheelchair}
                onChange={(e) => setFormData({
                  ...formData,
                  specialServices: { ...formData.specialServices, wheelchair: e.target.checked }
                })}
                className="form-checkbox h-5 w-5 text-blue-500"
              />
              <span className="text-gray-700 font-medium">Wheelchair Accessible</span>
            </div>
          </div> */}

          {/* Additional Notes */}
          <div className="space-y-4">
            <label className="block text-xl font-semibold text-gray-700 flex items-center">
              <span className="mr-2">📝</span> Additional Notes/Descriptions
            </label>
            <textarea
              placeholder="Add any additional notes or descriptions"
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              className="p-3 border border-gray-300 rounded-lg w-full focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none"
              rows="4"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full bg-blue-500 text-white py-3 rounded-lg shadow-md font-semibold hover:bg-blue-600 transition duration-300"
            onClick={handleSubmit}
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
};

export default SubmitResources;
