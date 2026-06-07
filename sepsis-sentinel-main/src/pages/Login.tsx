import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { Shield, Mail, Lock, Chrome } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { Navigate } from "react-router-dom";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import SignUpForm from "@/components/auth/SignUpForm";

export default function Login() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuth();

  if (authLoading) return null;
  if (user) return <Navigate to="/dashboard" replace />;

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      navigate("/dashboard");
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/dashboard`,
      },
    });
    if (error) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  };

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      });
      if (error) throw error;
      toast({ title: "Reset email sent", description: "Check your email for the password reset link." });
      setShowForgotPassword(false);
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>
      <div className="w-full max-w-md space-y-6">
        {/* Logo */}
        <div className="flex flex-col items-center gap-3">
          <div className="p-3 rounded-2xl bg-primary/10 border border-primary/20 animate-float">
            <Shield className="h-8 w-8 text-primary" />
          </div>
          <div className="text-center">
            <h1 className="text-xl font-bold text-foreground tracking-tight">
              QUANTUM SEPSIS SHIELD
            </h1>
            <p className="text-xs text-muted-foreground mt-1">
              Authorized ICU Personnel Only
            </p>
          </div>
        </div>

        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-base">
              {showForgotPassword ? "Reset Password" : isLogin ? "Sign In" : "Create Account"}
            </CardTitle>
            <CardDescription className="text-xs">
              {showForgotPassword
                ? "Enter your email to receive a reset link"
                : isLogin
                ? "Access the ICU monitoring dashboard"
                : "Register as ICU staff — all fields required"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {showForgotPassword ? (
              <form onSubmit={handleForgotPassword} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="reset-email" className="text-xs">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="reset-email"
                      type="email"
                      placeholder="staff@hospital.org"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10 text-sm"
                      required
                    />
                  </div>
                </div>
                <Button type="submit" className="w-full text-sm" disabled={loading}>
                  {loading ? "Sending..." : "Send Reset Link"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className="w-full text-xs"
                  onClick={() => setShowForgotPassword(false)}
                >
                  Back to sign in
                </Button>
              </form>
            ) : isLogin ? (
              <>
                {/* Google SSO */}
                <Button variant="outline" className="w-full text-sm" onClick={handleGoogleLogin}>
                  <Chrome className="h-4 w-4 mr-2" />
                  Continue with Google
                </Button>

                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-border" />
                  </div>
                  <div className="relative flex justify-center text-xs">
                    <span className="bg-card px-2 text-muted-foreground">or</span>
                  </div>
                </div>

                <form onSubmit={handleLogin} className="space-y-3">
                  <div className="space-y-2">
                    <Label htmlFor="email" className="text-xs">Email</Label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="email"
                        type="email"
                        placeholder="staff@hospital.org"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="pl-10 text-sm"
                        required
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="password" className="text-xs">Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="password"
                        type="password"
                        placeholder="••••••••"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="pl-10 text-sm"
                        minLength={6}
                        required
                      />
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => setShowForgotPassword(true)}
                    className="text-xs text-primary hover:underline"
                  >
                    Forgot password?
                  </button>

                  <Button type="submit" className="w-full text-sm" disabled={loading}>
                    {loading ? "Signing in..." : "Sign In"}
                  </Button>
                </form>

                <div className="text-center flex items-center justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => setIsLogin(false)}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Need an account? Sign up
                  </button>
                  <span className="text-muted-foreground">·</span>
                  <button
                    type="button"
                    onClick={() => navigate("/register")}
                    className="text-xs text-primary hover:underline"
                  >
                    Register Hospital
                  </button>
                </div>
              </>
            ) : (
              <SignUpForm onSwitchToLogin={() => setIsLogin(true)} />
            )}
          </CardContent>
        </Card>

        <p className="text-center text-[10px] text-muted-foreground">
          Access restricted to authorized ICU personnel
        </p>
      </div>
    </div>
  );
}
