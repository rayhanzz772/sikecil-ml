import os
from flask import Flask, jsonify
from flask_cors import CORS

from routes.prediction_route import prediction_bp
from routes.prediction_route_v1 import prediction_v1_bp
from routes.prediction_route_v3 import prediction_v3_bp


# ==========================================================
# APP FACTORY
# ==========================================================

def create_app() -> Flask:
    """
    Flask application factory.

    Returns
    -------
    Flask
        Instance aplikasi yang sudah dikonfigurasi.
    """
    app = Flask(__name__)
    
    # ----------------------------------------------------------
    # Konfigurasi CORS
    # ----------------------------------------------------------
    CORS(app)  # Mengizinkan semua origin secara default

    # ----------------------------------------------------------
    # Konfigurasi
    # ----------------------------------------------------------
    app.config["JSON_SORT_KEYS"]    = False   # Pertahankan urutan key di response JSON
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    # ----------------------------------------------------------
    # Register Blueprint
    # ----------------------------------------------------------
    app.register_blueprint(prediction_bp)
    app.register_blueprint(prediction_v1_bp)
    app.register_blueprint(prediction_v3_bp)

    @app.route("/")
    def health():
        return jsonify({"status": "ok"})

    # ----------------------------------------------------------
    # Global Error Handlers
    # ----------------------------------------------------------

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"success": False, "error": "Bad Request: " + str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "error": "Endpoint tidak ditemukan."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"success": False, "error": "Method tidak diizinkan untuk endpoint ini."}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"success": False, "error": "Internal Server Error."}), 500

    return app


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    app = create_app()

    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port       = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )
