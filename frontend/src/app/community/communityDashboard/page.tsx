'use client';
import { useAuth } from '../../../context/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function CommunityDashboard() {
  const { communityUser } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!communityUser) {
      router.push('/community/login');
    }
  }, [communityUser, router]);

  if (!communityUser) return <p>Redirecting...</p>;

  return <div>Welcome to the Provider Dashboard</div>;
}
