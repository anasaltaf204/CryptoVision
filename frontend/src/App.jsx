/**
 * App.jsx — added /profile protected route.
 */

import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthContextProvider } from "./context/AuthContext";
import Navbar         from "./components/Navbar";
import Footer         from "./components/Footer";
import ProtectedRoute from "./components/ProtectedRoute";
import Dashboard      from "./pages/Dashboard";
import CoinDetail     from "./pages/CoinDetail";
import Prediction     from "./pages/Prediction";
import LoginSignUp    from "./pages/LoginSignUp";
import Profile        from "./pages/Profile";

export default function App() {
    return (
        <AuthContextProvider>
            <Router>
                <div className="app-wrapper d-flex flex-column min-vh-100">
                    <Navbar />
                    <div className="flex-grow-1">
                        <Routes>
                            {/* Public */}
                            <Route path="/"             element={<Dashboard />} />
                            <Route path="/coin/:coinId" element={<CoinDetail />} />
                            <Route path="/auth"         element={<LoginSignUp />} />

                            {/* Protected */}
                            <Route path="/prediction" element={
                                <ProtectedRoute><Prediction /></ProtectedRoute>
                            }/>
                            <Route path="/profile" element={
                                <ProtectedRoute><Profile /></ProtectedRoute>
                            }/>

                            {/* Fallback */}
                            <Route path="*" element={<Navigate to="/" replace />} />
                        </Routes>
                    </div>
                    <Footer />
                </div>
            </Router>
        </AuthContextProvider>
    );
}
