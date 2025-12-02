import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Message } from '../types';
import { Bot, User as UserIcon, AlertCircle, Copy, Check } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
  codeBlockTheme?: 'light' | 'dark';
  showLineNumbers?: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message, 
  isStreaming,
  codeBlockTheme = 'dark',
  showLineNumbers = true
}) => {
  const isUser = message.role === 'user';
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = async () => {
    if (!message.text) return;
    
    try {
      await navigator.clipboard.writeText(message.text);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-6 group`}>
      <div 
        className={`flex max-w-[85%] md:max-w-[75%] gap-4 ${
          isUser ? 'flex-row-reverse' : 'flex-row'
        }`}
      >
        {/* Avatar */}
        <div className={`
          flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
          ${isUser ? 'bg-blue-600' : 'bg-gradient-to-br from-blue-500 to-purple-600'}
        `}>
          {isUser ? (
            <UserIcon size={16} className="text-white" />
          ) : (
            <Bot size={16} className="text-white" />
          )}
        </div>

        {/* Content Bubble */}
        <div className={`
          flex flex-col 
          ${isUser ? 'items-end' : 'items-start'} 
          min-w-0 flex-1
        `}>
          <div className="font-medium text-xs text-gray-400 mb-1 px-1">
            {isUser ? 'You' : 'Gemini'}
          </div>
          
          <div 
            className={`
              relative px-5 py-3.5 rounded-2xl text-sm md:text-base shadow-sm
              ${isUser 
                ? 'bg-[#1e1f24] text-gray-100 rounded-tr-sm border border-gray-800' 
                : 'bg-transparent text-gray-100 px-0 py-0 shadow-none'
              }
              ${message.isError ? 'border-red-500/50 bg-red-900/10' : ''}
            `}
          >
             {message.isError ? (
               <div className="flex items-center text-red-400 gap-2">
                 <AlertCircle size={16} />
                 <span>Error generating response. Please try again.</span>
               </div>
             ) : (
               <div className={`markdown-body ${isStreaming ? 'typing-cursor' : ''}`}>
                 <ReactMarkdown
                   urlTransform={(url) => url} // CRITICAL: Allow data: URLs for base64 images
                   components={{
                     code({node, inline, className, children, ...props}: any) {
                       const match = /language-(\w+)/.exec(className || '');
                       const isMultiLine = String(children).includes('\n');
                       
                       return !inline && (match || isMultiLine) ? (
                         <div className="relative group/code my-4 rounded-lg overflow-hidden">
                           <SyntaxHighlighter
                             {...props}
                             style={codeBlockTheme === 'dark' ? vscDarkPlus : oneLight}
                             language={match ? match[1] : ''}
                             PreTag="div"
                             showLineNumbers={showLineNumbers}
                             customStyle={{
                               margin: 0,
                               borderRadius: '0.5rem',
                               fontSize: '0.9em',
                               lineHeight: '1.5',
                             }}
                             wrapLongLines={true}
                           >
                             {String(children).replace(/\n$/, '')}
                           </SyntaxHighlighter>
                         </div>
                       ) : (
                         <code className={className} {...props}>
                           {children}
                         </code>
                       )
                     },
                     img({node, ...props}: any) {
                        return (
                          <div className="my-4">
                            <img 
                              {...props} 
                              className="max-w-full h-auto rounded-xl border border-gray-700 shadow-md block"
                              loading="lazy" 
                              alt={props.alt || "Generated Image"}
                            />
                          </div>
                        );
                     }
                   }}
                 >
                   {message.text}
                 </ReactMarkdown>
               </div>
             )}
          </div>

          {/* Action Bar for AI Messages */}
          {!isUser && !message.isError && message.text && (
             <div className="flex items-center gap-2 mt-2 px-1">
               <button
                 onClick={handleCopy}
                 className="flex items-center gap-1.5 px-2 py-1 text-xs font-medium text-gray-500 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-colors"
                 title="Copy to clipboard"
                 aria-label="Copy response to clipboard"
               >
                 {isCopied ? (
                   <Check size={14} className="text-emerald-400" />
                 ) : (
                   <Copy size={14} />
                 )}
                 <span className={isCopied ? "text-emerald-400" : ""}>
                   {isCopied ? 'Copied' : 'Copy'}
                 </span>
               </button>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};