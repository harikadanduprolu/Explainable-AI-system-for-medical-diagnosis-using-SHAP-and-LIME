import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import '../styles/Auth.css';

const Signup = () => {
  const navigate = useNavigate();
  const { signup } = useAuth();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    fullName: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [passwordRequirements, setPasswordRequirements] = useState({
    length: false,
    uppercase: false,
    lowercase: false,
    digit: false
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });

    if (name === 'password') {
      setPasswordRequirements({
        length: value.length >= 8,
        uppercase: /[A-Z]/.test(value),
        lowercase: /[a-z]/.test(value),
        digit: /\d/.test(value)
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!Object.values(passwordRequirements).every(req => req)) {
      setError('Password does not meet requirements');
      return;
    }

    setLoading(true);

    try {
      await signup(formData.username, formData.email, formData.password, formData.fullName);
      navigate('/clinical');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card signup-card">
        <div className="auth-header">
          <div className="auth-icon">🏥</div>
          <h1>Medical AI System</h1>
          <p>Create your account</p>
        </div>

        <div className="auth-body">
          {error && (
            <div className="alert alert-error">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                type="text"
                id="username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Choose a username"
                minLength="3"
                maxLength="50"
                pattern="[a-zA-Z0-9_-]+"
                required
              />
              <small>3-50 characters, alphanumeric with _ or -</small>
            </div>

            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="your.email@example.com"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="fullName">Full Name (Optional)</label>
              <input
                type="text"
                id="fullName"
                name="fullName"
                value={formData.fullName}
                onChange={handleChange}
                placeholder="Your full name"
                maxLength="100"
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Create a strong password"
                minLength="8"
                required
              />
              <div className="password-requirements">
                <strong>Password must contain:</strong>
                <ul>
                  <li className={passwordRequirements.length ? 'requirement-met' : 'requirement-unmet'}>
                    At least 8 characters
                  </li>
                  <li className={passwordRequirements.uppercase ? 'requirement-met' : 'requirement-unmet'}>
                    One uppercase letter
                  </li>
                  <li className={passwordRequirements.lowercase ? 'requirement-met' : 'requirement-unmet'}>
                    One lowercase letter
                  </li>
                  <li className={passwordRequirements.digit ? 'requirement-met' : 'requirement-unmet'}>
                    One number
                  </li>
                </ul>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Re-enter your password"
                required
              />
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? (
                <>
                  <span className="spinner"></span>
                  <span>Creating Account...</span>
                </>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div className="auth-link">
            Already have an account? <Link to="/login">Login</Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Signup;
