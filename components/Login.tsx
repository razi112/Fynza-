import React, { useState } from 'react';
import { X } from 'lucide-react';

interface LoginProps {
  onLogin: (email: string) => void;
}

export const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);

  const handleEmailLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    setLoading(true);
    // Simulate network request
    setTimeout(() => {
        onLogin(email);
        setLoading(false);
    }, 1000);
  };

  const handleSocialLogin = (provider: string) => {
      // Specific handling for Google as requested to redirect to Google Accounts
      if (provider === 'Google') {
        window.location.href = 'https://accounts.google.com';
        return;
      }

      setLoading(true);
      // Simulate network request with a realistic delay
      setTimeout(() => {
          // Provide realistic mock emails based on provider
          const mockEmail = `${provider.toLowerCase()}@example.com`;
          onLogin(mockEmail);
          setLoading(false);
      }, 1000);
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0f1115] p-4 font-sans">
       {/* Card Container */}
       <div className="w-full max-w-[480px] bg-[#1e1f20] rounded-[24px] p-6 md:p-10 relative shadow-2xl border border-gray-800/50">
          
          {/* Close Button - Visual only for root login */}
          <button className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors p-2 rounded-full hover:bg-white/10">
            <X size={24} />
          </button>

          {/* Header */}
          <div className="text-center mb-8 mt-2">
            <h1 className="text-3xl font-normal text-white mb-3">Log in or sign up</h1>
            <p className="text-[#e3e3e3] text-sm leading-relaxed max-w-[320px] mx-auto">
              You’ll get smarter responses and can upload files, images, and more.
            </p>
          </div>

          {/* Social Buttons */}
          <div className="space-y-3 mb-8">
            {/* Google */}
            <button 
              type="button"
              onClick={() => handleSocialLogin('Google')}
              className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-[#5f6368] hover:bg-[#303134] transition-colors group bg-transparent"
            >
               <svg className="w-5 h-5" viewBox="0 0 24 24">
                 <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                 <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                 <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                 <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
               </svg>
               <span className="text-white font-medium text-sm">Continue with Google</span>
            </button>

            {/* Apple */}
            <button 
               type="button"
               onClick={() => handleSocialLogin('Apple')}
               className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-[#5f6368] hover:bg-[#303134] transition-colors bg-transparent"
            >
               <svg className="w-5 h-5 fill-white" viewBox="0 0 24 24">
                  <path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.74s2.22-.91 3.69-.76c.49.02 2.13.18 3.19 1.63-2.62 1.55-2.2 5.66.45 6.78-.39 1.34-1.22 3.1-2.41 4.58zM12.03 5.32c.6-1.58 2.65-2.5 4.38-2.31.25 1.9-1.92 3.99-4.38 2.31z"/>
               </svg>
               <span className="text-white font-medium text-sm">Continue with Apple</span>
            </button>
            
            {/* Microsoft */}
            <button 
               type="button"
               onClick={() => handleSocialLogin('Microsoft')}
               className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-[#5f6368] hover:bg-[#303134] transition-colors bg-transparent"
            >
               <svg className="w-5 h-5" viewBox="0 0 23 23">
                  <path fill="#F25022" d="M1 1h10v10H1z"/>
                  <path fill="#00A4EF" d="M12 1h10v10H12z"/>
                  <path fill="#7FBA00" d="M1 12h10v10H1z"/>
                  <path fill="#FFB900" d="M12 12h10v10H12z"/>
               </svg>
               <span className="text-white font-medium text-sm">Continue with Microsoft</span>
            </button>

             {/* GitHub (Added per previous request) */}
             <button 
               type="button"
               onClick={() => handleSocialLogin('GitHub')}
               className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-[#5f6368] hover:bg-[#303134] transition-colors bg-transparent"
            >
               <svg className="w-5 h-5 fill-white" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
               </svg>
               <span className="text-white font-medium text-sm">Continue with GitHub</span>
            </button>

            {/* Phone */}
             <button 
               type="button"
               onClick={() => handleSocialLogin('Phone')}
               className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-[#5f6368] hover:bg-[#303134] transition-colors bg-transparent"
            >
               <svg className="w-5 h-5 stroke-white" fill="none" viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                   <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
               </svg>
               <span className="text-white font-medium text-sm">Continue with phone</span>
            </button>
          </div>

          {/* OR Divider */}
          <div className="relative flex items-center justify-center mb-8">
            <div className="absolute w-full h-px bg-[#5f6368]/50"></div>
            <span className="relative bg-[#1e1f20] px-4 text-xs text-gray-300 font-medium tracking-widest">OR</span>
          </div>

          {/* Email Form */}
          <form onSubmit={handleEmailLogin}>
            <div className="mb-10">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-transparent border border-[#5f6368] text-white rounded-[16px] px-4 py-4 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 transition-all placeholder-gray-500 text-base"
                placeholder="Email address"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`
                w-full bg-white text-[#1f1f1f] font-medium rounded-full py-3.5
                hover:bg-gray-200 transition-colors
                ${loading ? 'opacity-70 cursor-not-allowed' : ''}
              `}
            >
              {loading ? 'Continuing...' : 'Continue'}
            </button>
          </form>

       </div>
    </div>
  );
};