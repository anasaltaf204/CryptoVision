/**
 * components/Navbar.jsx
 *
 * Changes:
 *  - Avatar now shows photoURL image if present, initials fallback otherwise.
 *  - User display name pulled from Redux (selectUserProfile) first, then
 *    falls back to Firebase Auth currentUser.displayName for instant render
 *    before the Firestore load completes.
 *  - Added clickable avatar/name that navigates to /profile.
 *  - Loading skeleton shimmer shown while profile is loading.
 */

import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { useAuth } from "../context/AuthContext";
import { selectUserProfile, selectUserStatus } from "../store/slices/userSlice";
import "../Styles/Navbar.css";

function NavAvatar({ photoURL, displayName }) {
    if (photoURL) {
        return (
            <img
                src={photoURL}
                alt={displayName}
                className="user-avatar user-avatar--photo"
            />
        );
    }
    return (
        <div className="user-avatar">
            {displayName?.[0]?.toUpperCase() || "U"}
        </div>
    );
}

export default function Navbar() {
    const { currentUser, logout } = useAuth();
    const profile     = useSelector(selectUserProfile);
    const userStatus  = useSelector(selectUserStatus);
    const navigate    = useNavigate();
    const [isCollapsed, setIsCollapsed] = useState(true);

    // Prioritise Firestore profile name; fall back to Auth displayName
    // so the navbar renders immediately on page load with the cached Auth value
    const displayName = profile?.username || currentUser?.displayName || "User";
    const photoURL    = profile?.photoURL  || currentUser?.photoURL   || null;
    const isLoading   = userStatus === "loading";

    const handleLogout = async () => {
        try {
            await logout();
            navigate("/");
            setIsCollapsed(true);
        } catch (err) {
            console.error("Logout failed:", err);
        }
    };

    const handleNavLinkClick  = () => setIsCollapsed(true);
    const handleAuthClick     = () => { navigate("/auth");    setIsCollapsed(true); };
    const handleProfileClick  = () => { navigate("/profile"); setIsCollapsed(true); };

    return (
        <nav className="navbar navbar-expand-lg navbar-dark navbar-enhanced px-lg-5">
            <div className="container-fluid">
                <Link className="navbar-brand" to="/" onClick={handleNavLinkClick}>
                    CryptoVision
                </Link>

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
                        <span></span><span></span><span></span>
                    </div>
                </button>

                <div className="collapse navbar-collapse" id="navbarNav">
                    <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
                        <li className="nav-item">
                            <Link className="nav-link" to="/" onClick={handleNavLinkClick}>
                                Dashboard
                            </Link>
                        </li>
                        {currentUser && (
                            <li className="nav-item">
                                <Link className="nav-link" to="/prediction" onClick={handleNavLinkClick}>
                                    Prediction
                                </Link>
                            </li>
                        )}
                    </ul>

                    {!currentUser ? (
                        <div className="auth-buttons">
                            <button className="btn-login py-2 px-4" onClick={handleAuthClick}>Login</button>
                            <button className="btn-signup py-2 px-4" onClick={handleAuthClick}>Sign Up</button>
                        </div>
                    ) : (
                        <div className="auth-buttons">
                            {/* Clickable profile chip */}
                            <button
                                className={`user-profile-btn ${isLoading ? "user-profile-btn--loading" : ""}`}
                                onClick={handleProfileClick}
                                title="View profile"
                            >
                                {isLoading ? (
                                    <div className="user-avatar user-avatar--skeleton" />
                                ) : (
                                    <NavAvatar photoURL={photoURL} displayName={displayName} />
                                )}
                                <span className="user-name">
                                    {isLoading ? (
                                        <span className="skeleton-text" />
                                    ) : (
                                        displayName
                                    )}
                                </span>
                            </button>
                            <button className="btn-logout py-2 px-3" onClick={handleLogout}>
                                Logout
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </nav>
    );
}
