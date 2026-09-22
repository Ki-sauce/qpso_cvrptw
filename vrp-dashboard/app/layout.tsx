import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QPSO Dispatch Platform",
  description: "Enterprise Route Optimizer",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark" style={{ colorScheme: "dark" }}>
      <body className="font-sans antialiased bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
