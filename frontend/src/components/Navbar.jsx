import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import "../Styles/Navbar.css";



export default function Navbar() {
    const { currentUser, logout } = useAuth();
    const navigate = useNavigate();
    const [isCollapsed, setIsCollapsed] = useState(true);

    const handleLogout = async () => {
        try {
            await logout();
            navigate("/");
            setIsCollapsed(true);
        } catch (error) {
            console.error("Logout failed:", error);
        }
    };

    const handleNavLinkClick = () => {
        setIsCollapsed(true);
    };

    const handleAuthClick = () => {
        navigate("/auth");
        setIsCollapsed(true);
    };

    return (
        <>
            <nav className="navbar navbar-expand-lg navbar-dark navbar-enhanced px-lg-5">
                <div className="container-fluid">
                    {/* Brand */}
                    <Link className="navbar-brand" to="/" onClick={handleNavLinkClick}>
                        CryptoVision
                    </Link>

                    {/* Toggler */}
                    <button
                        className="navbar-toggler custom-toggler"
                        type="button"
                        data-bs-toggle="collapse"
                        data-bs-target="#navbarNav"
                        aria-controls="navbarNav"
                        aria-expanded={!isCollapsed}
                        aria-label="Toggle navigation"
                        onClick={() => setIsCollapsed(!isCollapsed)}
                    >
                        <div className="toggler-icon">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </button>

                    {/* Collapsible Navigation */}
                    <div className="collapse navbar-collapse" id="navbarNav">
                        <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
                            <li className="nav-item">
                                <Link 
                                    className="nav-link" 
                                    to="/"
                                    onClick={handleNavLinkClick}
                                >
                                        Dashboard
                                </Link>
                            </li>
                            {currentUser && (
                                <li className="nav-item">
                                    <Link 
                                        className="nav-link" 
                                        to="/prediction"
                                        onClick={handleNavLinkClick}
                                    >
                                            Prediction
                                    </Link>
                                </li>
                            )}
                        </ul>

                        {/* Auth Section */}
                        {!currentUser ? (
                            <div className="auth-buttons">
                                <button 
                                    className="btn-login py-2 px-4"
                                    onClick={handleAuthClick}
                                >
                                    Login
                                </button>
                                <button 
                                    className="btn-signup py-2 px-4"
                                    onClick={handleAuthClick}
                                >
                                    Sign Up
                                </button>
                            </div>
                        ) : (
                            <div className="auth-buttons">
                                <div className="user-profile">
                                    <div className="user-avatar">
                                        {currentUser.displayName?.[0]?.toUpperCase() || "U"}
                                    </div>
                                    <span className="user-name">
                                        {currentUser.displayName || "User"}
                                    </span>
                                </div>
                                <button 
                                    className="btn-logout py-2 px-3"
                                    onClick={handleLogout}
                                >
                                        Logout
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </nav>
        </>
    );
}
