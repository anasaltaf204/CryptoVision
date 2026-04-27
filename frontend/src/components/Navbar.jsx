/**
 * components/Navbar.jsx
 *
 * Bug fixes in this version:
 *
 * BUG 1 — Split brain between React state and Bootstrap JS:
 *   The original used BOTH `data-bs-toggle="collapse"` (Bootstrap JS) AND
 *   React state (isCollapsed) to control the menu. Bootstrap JS directly
 *   manipulates the DOM class on #navbarNav, while React state controlled
 *   aria-expanded and the X animation. They fought each other: Bootstrap
 *   opened the menu but React thought it was still closed (and vice versa).
 *   FIX: Remove data-bs-toggle / data-bs-target entirely. React owns the
 *   open/close state exclusively. We manually toggle a CSS class on the
 *   collapse div using the `show` class Bootstrap expects.
 *
 * BUG 2 — X animation CSS selector was wrong:
 *   `.custom-toggler:not(.collapsed)` targets the button when it does NOT
 *   have the class "collapsed". But Bootstrap adds "collapsed" to the button
 *   when the menu is CLOSED — so the selector was inverted AND relied on
 *   Bootstrap JS being present. Since we now manage state in React, we use
 *   a data attribute `data-open="true/false"` that we set ourselves.
 *   FIX: CSS selectors updated to `[data-open="true"]`.
 *
 * BUG 3 — isCollapsed initial value logic was inverted in aria-expanded:
 *   `aria-expanded={!isCollapsed}` was correct but misleading. Now that
 *   we use `isOpen` (true = open) the aria value is simply `{isOpen}`.
 */

import React, { useState, useRef, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
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
    const profile    = useSelector(selectUserProfile);
    const userStatus = useSelector(selectUserStatus);
    const navigate   = useNavigate();
    const location   = useLocation();
    const [isOpen, setIsOpen] = useState(false);
    const navRef     = useRef(null);

    const displayName = profile?.username || currentUser?.displayName || "User";
    const photoURL    = profile?.photoURL  || currentUser?.photoURL   || null;
    const isLoading   = userStatus === "loading";

    // Close menu on route change
    useEffect(() => {
        setIsOpen(false);
    }, [location.pathname]);

    // Close menu when clicking outside
    useEffect(() => {
        if (!isOpen) return;
        function handleClickOutside(e) {
            if (navRef.current && !navRef.current.contains(e.target)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [isOpen]);

    const close = () => setIsOpen(false);

    const handleLogout = async () => {
        try {
            await logout();
            navigate("/");
            close();
        } catch (err) {
            console.error("Logout failed:", err);
        }
    };

    const handleProfileClick = () => {
        navigate("/profile");
        close();
    };

    const handleAuthClick = () => {
        navigate("/auth");
        close();
    };

    return (
        <nav
            ref={navRef}
            className="navbar navbar-expand-lg navbar-dark navbar-enhanced px-lg-5"
        >
            <div className="container-fluid">
                {/* Brand */}
                <Link className="navbar-brand" to="/" onClick={close}>
                    CryptoVision
                </Link>

                {/* Hamburger — React-controlled, NO data-bs-toggle */}
                <button
                    className="navbar-toggler custom-toggler"
                    type="button"
                    aria-controls="navbarNav"
                    aria-expanded={isOpen}
                    aria-label="Toggle navigation"
                    data-open={isOpen}
                    onClick={() => setIsOpen((prev) => !prev)}
                >
                    <div className="toggler-icon">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </button>

                {/*
                  Add Bootstrap's `show` class manually based on React state.
                  This gives us the collapse animation without Bootstrap JS
                  fighting React over the DOM.
                */}
                <div
                    className={`collapse navbar-collapse ${isOpen ? "show" : ""}`}
                    id="navbarNav"
                >
                    <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
                        <li className="nav-item">
                            <Link className="nav-link" to="/" onClick={close}>
                                Dashboard
                            </Link>
                        </li>
                        {currentUser && (
                            <li className="nav-item">
                                <Link className="nav-link" to="/prediction" onClick={close}>
                                    Prediction
                                </Link>
                            </li>
                        )}
                    </ul>

                    {!currentUser ? (
                        <div className="auth-buttons">
                            <button className="btn-login py-2 px-4" onClick={handleAuthClick}>
                                Login
                            </button>
                            <button className="btn-signup py-2 px-4" onClick={handleAuthClick}>
                                Sign Up
                            </button>
                        </div>
                    ) : (
                        <div className="auth-buttons">
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
                                    {isLoading
                                        ? <span className="skeleton-text" />
                                        : displayName
                                    }
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
