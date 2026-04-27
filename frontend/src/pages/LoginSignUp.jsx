import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
    faFacebookF,
    faGoogle,
    faLinkedinIn
} from "@fortawesome/free-brands-svg-icons";
import "../Styles/LoginSignUp.css";

function LoginSignUp() {
    const { login, signup } = useAuth();
    const navigate = useNavigate();
    const [isSignup, setIsSignup] = useState(false);
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        try {
            if (isSignup) {
                if (password !== confirmPassword) {
                    setError("Passwords do not match");
                    setLoading(false);
                    return;
                }
                if (password.length < 6) {
                    setError("Password must be at least 6 characters");
                    setLoading(false);
                    return;
                }
                if (!username.trim()) {
                    setError("Username is required");
                    setLoading(false);
                    return;
                }
                await signup(email, password, username);
                setSuccess("Signup successful! Redirecting...");
                setTimeout(() => navigate('/prediction'), 1500);
            } else {
                await login(email, password);
                setSuccess("Login successful! Redirecting...");
                setTimeout(() => navigate('/prediction'), 1500);
            }
        } catch (err) {
            setError(err.message || "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const toggleMode = () => {
        setIsSignup(!isSignup);
        setError('');
        setSuccess('');
        setEmail('');
        setPassword('');
        setConfirmPassword('');
        setUsername('');
    };

    return (
        <>
            <div className="auth-container">
                <div className="auth-wrapper">
                    <div className="auth-grid">
                        {/* Form Container */}
                        <div className="auth-form-container">
                            <h1 className="auth-title">
                                {isSignup ? '✨ Create Account' : '🔐 Welcome Back'}
                            </h1>
                            <p className="auth-subtitle">
                                {isSignup 
                                    ? 'Join us to access AI predictions' 
                                    : 'Sign in to your account'}
                            </p>

                            {error && <div className="error-message">{error}</div>}
                            {success && <div className="success-message">{success}</div>}

                            <form onSubmit={handleSubmit}>
                                {isSignup && (
                                    <div className="form-group">
                                        <label className="form-label">Username</label>
                                        <input
                                            type="text"
                                            className="form-input"
                                            placeholder="Choose a username"
                                            value={username}
                                            onChange={(e) => setUsername(e.target.value)}
                                            disabled={loading}
                                        />
                                    </div>
                                )}

                                <div className="form-group">
                                    <label className="form-label">Email Address</label>
                                    <input
                                        type="email"
                                        className="form-input"
                                        placeholder="Enter your email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        disabled={loading}
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Password</label>
                                    <input
                                        type="password"
                                        className="form-input"
                                        placeholder="Enter your password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        disabled={loading}
                                        required
                                    />
                                </div>

                                {isSignup && (
                                    <div className="form-group">
                                        <label className="form-label">Confirm Password</label>
                                        <input
                                            type="password"
                                            className="form-input"
                                            placeholder="Confirm your password"
                                            value={confirmPassword}
                                            onChange={(e) => setConfirmPassword(e.target.value)}
                                            disabled={loading}
                                        />
                                    </div>
                                )}

                                <div className="form-group">
                                    <button
                                        type="submit"
                                        className={`form-button ${loading ? 'loading' : ''}`}
                                        disabled={loading}
                                    >
                                        {loading ? 'Processing...' : isSignup ? 'Create Account' : 'Sign In'}
                                    </button>
                                </div>
                            </form>

                            <div className="form-toggle">
                                <span className="form-toggle-text">
                                    {isSignup 
                                        ? 'Already have an account? ' 
                                        : "Don't have an account? "}
                                </span>
                                <span 
                                    className="form-toggle-link"
                                    onClick={toggleMode}
                                >
                                    {isSignup ? 'Sign In' : 'Sign Up'}
                                </span>
                            </div>

                            <div className="social-login">
                                <p className="social-text">Or {isSignup ? 'sign up' : 'sign in'} with</p>
                                <div className="social-icons">
                                    <div className="social-icon" title="Facebook">
                                        <FontAwesomeIcon icon={faFacebookF} />
                                    </div>
                                    <div className="social-icon" title="Google">
                                        <FontAwesomeIcon icon={faGoogle} />
                                    </div>
                                    <div className="social-icon" title="LinkedIn">
                                        <FontAwesomeIcon icon={faLinkedinIn} />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Info Container */}
                        <div className="auth-info-container">
                            <div>
                                <h2 className="info-title">
                                    {isSignup ? '👋 Welcome!' : '🚀 Ready?'}
                                </h2>
                                <p className="info-text">
                                    {isSignup
                                        ? 'Join thousands of traders using our AI-powered crypto prediction engine. Get started today and make smarter investment decisions.'
                                        : 'Access powerful AI predictions and market analysis. Sign in to unlock advanced features and real-time insights.'}
                                </p>
                                <button className="info-button" onClick={toggleMode}>
                                    {isSignup ? 'Already with us?' : 'New here?'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}

export default LoginSignUp;
