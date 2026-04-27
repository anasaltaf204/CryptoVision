/**
 * pages/Profile.jsx
 *
 * User profile page — thin orchestrator.
 * All data operations are in Redux (userSlice) and userService.
 * AvatarUpload and ProfileForm are isolated, reusable components.
 *
 * Flow:
 *  1. On mount, profile is already in Redux (loaded by AuthContext on login).
 *  2. User edits form fields → ProfileForm calls onSubmit(changes).
 *  3. User uploads avatar → handleAvatarUpload stores the data-URL and marks
 *     it as a pending change, submitted with the form.
 *  4. saveUserProfile thunk fires → optimistic merge in Redux on success.
 *  5. Success/error banners rendered inside ProfileForm.
 */

import React, { useCallback, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate }  from "react-router-dom";
import { useAuth }      from "../context/AuthContext";
import {
    saveUserProfile,
    resetSaveStatus,
    selectUserProfile,
    selectUserStatus,
    selectUserSaveState,
} from "../store/slices/userSlice";
import AvatarUpload from "../components/profile/AvatarUpload";
import ProfileForm  from "../components/profile/ProfileForm";
import "../Styles/Profile.css";

export default function Profile() {
    const dispatch   = useDispatch();
    const navigate   = useNavigate();
    const { currentUser } = useAuth();

    const profile   = useSelector(selectUserProfile);
    const status    = useSelector(selectUserStatus);
    const { saveStatus, saveError } = useSelector(selectUserSaveState);

    // Pending avatar data-URL (set before form submit)
    const [pendingAvatar, setPendingAvatar] = useState(null);

    // Auto-dismiss "saved" banner after 3 s
    useEffect(() => {
        if (saveStatus === "saved") {
            const t = setTimeout(() => dispatch(resetSaveStatus()), 3000);
            return () => clearTimeout(t);
        }
    }, [saveStatus, dispatch]);

    // Redirect if not logged in
    useEffect(() => {
        if (!currentUser) navigate("/auth", { replace: true });
    }, [currentUser, navigate]);

    const handleAvatarUpload = useCallback((dataUrl) => {
        setPendingAvatar(dataUrl);
    }, []);

    const handleFormSubmit = useCallback((changes) => {
        const updates = { ...changes };
        if (pendingAvatar) {
            updates.photoURL = pendingAvatar;
            setPendingAvatar(null);
        }
        dispatch(saveUserProfile({ uid: currentUser.uid, updates }));
    }, [dispatch, currentUser, pendingAvatar]);

    const displayName = profile?.username || currentUser?.displayName || "User";
    const photoURL    = pendingAvatar || profile?.photoURL || currentUser?.photoURL;

    if (status === "loading") {
        return (
            <div className="profile-page">
                <div className="container py-5">
                    <div className="profile-skeleton">
                        <div className="profile-skeleton__avatar" />
                        <div className="profile-skeleton__lines">
                            <div className="profile-skeleton__line profile-skeleton__line--wide" />
                            <div className="profile-skeleton__line profile-skeleton__line--medium" />
                            <div className="profile-skeleton__line profile-skeleton__line--short" />
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="profile-page">
            <div className="profile-page__hero">
                <div className="profile-page__hero-bg" />
                <div className="container">
                    <div className="profile-header">
                        <AvatarUpload
                            currentURL={photoURL}
                            displayName={displayName}
                            onUpload={handleAvatarUpload}
                            size={112}
                        />
                        <div className="profile-header__info">
                            <h1 className="profile-header__name">{displayName}</h1>
                            <p className="profile-header__email">
                                {profile?.email || currentUser?.email}
                            </p>
                            {profile?.bio && (
                                <p className="profile-header__bio">{profile.bio}</p>
                            )}
                            {profile?.createdAt && (
                                <p className="profile-header__joined">
                                    Member since{" "}
                                    {new Date(profile.createdAt.toDate?.() || profile.createdAt)
                                        .toLocaleDateString("en-US", { month: "long", year: "numeric" })}
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <div className="container py-4">
                <div className="profile-layout">
                    {/* Left: edit form */}
                    <div className="profile-card">
                        <h2 className="profile-card__title">Edit Profile</h2>
                        <ProfileForm
                            initialValues={{
                                username: profile?.username || currentUser?.displayName || "",
                                email:    profile?.email    || currentUser?.email        || "",
                                bio:      profile?.bio      || "",
                            }}
                            onSubmit={handleFormSubmit}
                            saveStatus={saveStatus}
                            saveError={saveError}
                        />
                    </div>

                    {/* Right: stats sidebar */}
                    <div className="profile-sidebar">
                        <div className="profile-card profile-card--stats">
                            <h3 className="profile-card__title">Account</h3>
                            <ul className="profile-stats">
                                <li className="profile-stats__item">
                                    <span className="profile-stats__label">Status</span>
                                    <span className="profile-stats__value profile-stats__value--active">
                                        ● Active
                                    </span>
                                </li>
                                <li className="profile-stats__item">
                                    <span className="profile-stats__label">UID</span>
                                    <span className="profile-stats__value profile-stats__value--mono">
                                        {currentUser?.uid?.slice(0, 12)}…
                                    </span>
                                </li>
                                <li className="profile-stats__item">
                                    <span className="profile-stats__label">Auth</span>
                                    <span className="profile-stats__value">
                                        {currentUser?.providerData?.[0]?.providerId || "email"}
                                    </span>
                                </li>
                            </ul>
                        </div>

                        {pendingAvatar && (
                            <div className="profile-card profile-card--pending">
                                <p className="profile-pending-notice">
                                    📷 New photo ready — save the form to apply it.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
