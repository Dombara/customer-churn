// export default function ResourceProviderLayout({ children }) {
//     return (
//       <div>
//         <header>Resource Provider Dashboard</header>
//         <main>{children}</main>
//       </div>
//     );
//   }
  
export default function ResourceProviderLayout({ children }) {
  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      {/* <header className="bg-black text-white py-4 shadow-md">
        <div className="max-w-7xl mx-auto px-4">
          <h1 className="text-2xl font-semibold">Resource Provider Dashboard</h1>
        </div>
      </header> */}

      {/* Main Content */}
      <main className="py-8">
        <div className="max-w-3xl mx-auto px-4">
          {children}
        </div>
      </main>
    </div>
  );
}
