'use client';
import { createContext, useContext, useState, useEffect } from 'react';

// Create the Auth Context
const AuthContext = createContext();

// Create a custom hook to use the Auth Context
export const useAuth = () => {
  return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);

    useEffect(()=>{
      const storedUser = localStorage.getItem('user');
      if(storedUser){
        setUser(JSON.parse(storedUser));
      }
    },[])
  
    const loginUser = (userData) => {
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
    }
    const logoutUser = () => {
      try{
        setUser(null);
        localStorage.removeItem('user');
      }
      catch (error) {
        console.warn("Logout failed:", error);
        alert("Failed to logout.");
      }
    };

    return (
        <AuthContext.Provider value={{user, loginUser, logoutUser}}>
          {children}
        </AuthContext.Provider>
      );
    };
